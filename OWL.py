from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib import pyplot as plt
import cv2

SAM2_PATH = "/home/mverghese/sam2/checkpoints/sam2.1_hiera_large.pt"
np.random.seed(3)

def visualize_mask(img, mask, alpha=0.7, color=(255, 0, 0)):
	mask_im = np.zeros(img.shape,dtype=np.uint8)
	mask_im[:,:,:] = color
	mask_im = cv2.bitwise_and(mask_im,mask_im,mask=mask.astype(np.uint8))
	disp_im = cv2.addWeighted(img, alpha , mask_im, 1-alpha, 0)
	return disp_im

class OWLDetector:
	def __init__(self, model_name="google/owlv2-base-patch16-ensemble", sam_checkpoint=SAM2_PATH, device=None):
		self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.processor = Owlv2Processor.from_pretrained(model_name)
		self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
		if sam_checkpoint:
			if "large" in sam_checkpoint:
				model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
			elif "base_plus" in sam_checkpoint:
				model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
			elif "small" in sam_checkpoint:
				model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
			elif "tiny" in sam_checkpoint:
				model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
			else:
				raise ValueError("Invalid SAM model checkpoint")
			self.sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint, device=self.device))
		else:
			self.sam_predictor = None

			

		self.colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
	
	def inference(self, image, text_labels):
		if isinstance(image, np.ndarray):
			if len(image.shape) == 3:
				image = Image.fromarray(image)
				target_sizes = torch.tensor([(image.height, image.width)])
			elif len(image.shape) == 4:
				image = [Image.fromarray(im) for im in image]
				target_sizes = torch.tensor([(im.height, im.width) for im in image])
		elif isinstance(image, list):
			image = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in image]
			target_sizes = torch.tensor([(im.height, im.width) for im in image])
		else:
			target_sizes = torch.tensor([(image.height, image.width)])
		inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
		outputs = self.model(**inputs)
		# target_sizes = torch.tensor([(image.height, image.width)])
		results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
		return results

	def mask_inference(self, image, text_labels):
		if isinstance(image, np.ndarray) and len(image.shape) == 4:
			image = [im for im in image]
		results = self.inference(image, text_labels)
		# print(results)
		labels = [results[i]["labels"].cpu().detach().numpy() for i in range(len(results))]
		scores = [results[i]["scores"].cpu().detach().numpy() for i in range(len(results))]
		boxes = [results[i]["boxes"].cpu().detach().tolist() for i in range(len(results))]
		if isinstance(text_labels[0], str):
			num_classes = [len(text_labels)]
		else:
			num_classes = [len(text_labels[i]) for i in range(len(text_labels))]

		all_boxes = []
		all_scores = []
		for i in range(len(results)):
			class_indices = [np.where(labels[i] == j)[0] for j in range(num_classes[i])]
			class_scores = [scores[i][indices] for indices in class_indices]
			# import pdb; pdb.set_trace()
			all_scores.append([np.max(class_score) if len(class_score) > 0 else 0 for class_score in class_scores])
			best_boxes = [boxes[i][class_indices[j][np.argmax(class_score)]] if len(class_score) > 0 else [0,0,0,0] for j, class_score in enumerate(class_scores)]
			best_boxes = [[int(round(i, 0)) for i in box] for box in best_boxes]
			all_boxes.append(best_boxes)
		if isinstance(image, list):
			best_masks = []
			for i in range(len(image)):
				if np.all(all_boxes[i] == 0):
					best_masks.append(np.zeros((image[i].shape[0], image[i].shape[1])))
					continue
				self.sam_predictor.set_image(image[i])
				masks, scores, _ = self.sam_predictor.predict(box = all_boxes[i])
				if len(scores.shape) == 1:
					scores = scores.reshape(1, -1)
					# add a singleton dimension to axis 0 of the masks
					masks = masks.reshape(1, masks.shape[0], masks.shape[1], masks.shape[2])
				class_mask_indices = np.argmax(scores, axis=1)
				all_scores[i] *= np.max(scores, axis=1)
				best_masks.append(masks[range(num_classes[i]),class_mask_indices.tolist(), :,:])
			return best_masks, all_scores
		else:
			if np.all(all_boxes[0] == 0):
				return np.zeros((image.shape[0], image.shape[1]))
			self.sam_predictor.set_image(image)
			masks, scores, _ = self.sam_predictor.predict(box = all_boxes[0])
			best_mask = masks[np.argmax(scores)]
			all_scores[0] *= np.max(scores)
			
			return best_mask, all_scores[0]
	
	def batched_mask_inference(self, images, text_labels, batch_size = 20):
		all_masks = []
		all_scores = []
		for i in range(0, len(images), batch_size):
			batch_images = images[i:i + batch_size]
			batch_labels = text_labels[i:i + batch_size]
			masks, scores = self.mask_inference(batch_images, batch_labels)
			all_masks.extend(masks)
			all_scores.extend(scores)
		return all_masks, all_scores

	def generate_visualization(self, image, labels, results):
		for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
			box = [round(i, 2) for i in box.tolist()]
			color = self.colors[label % len(self.colors)]
			image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
			image = cv2.putText(image, f"{labels[label]}: {round(score.item(), 3)}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return image

	def get_obj_box(self, image, obj_label):
		if isinstance(image, list):
			obj_label = [obj_label] * len(image)
			is_batch = True
		else:
			obj_label = [obj_label]
			is_batch = False
		results = self.inference(image, obj_label)
		boxes = []

		scores = [results[i]["scores"].cpu().detach() for i in range(len(results))]
		
		if len(scores) == 0:
			return np.zeros(4)
		for i in range(len(results)):
			if len(scores[i]) == 0:
				continue
			box = results[i]["boxes"][np.argmax(scores[i])].tolist()
			box = [round(i, 2) for i in box]
			boxes.append(box)
		
		if not is_batch:
			boxes = boxes[0]
		
		return boxes
	
	def get_obj_mask(self, image, obj_label):
		results = self.inference(image, [obj_label])
		scores = results[0]["scores"].cpu().detach()
		# print(scores)

		if len(scores) == 0:
			return np.zeros((image.shape[0], image.shape[1]))
		box = results[0]["boxes"][np.argmax(scores)].tolist()
		box = [int(i) for i in box]
		if self.sam_predictor is None:
			# set mask to be the same as the box
			mask = np.zeros((image.shape[0], image.shape[1]))
			mask[box[1]:box[3], box[0]:box[2]] = 1
			return mask
		if isinstance(image, Image.Image):
			image = np.array(image)
		self.sam_predictor.set_image(image)
		masks, scores, _ = self.sam_predictor.predict(box = [box])
		best_mask = masks[np.argmax(scores)]
		return best_mask
		

def get_sam_predictor(sam_checkpoint=SAM2_PATH, device=None):
	device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if "large" in sam_checkpoint:
		model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
	elif "base_plus" in sam_checkpoint:
		model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
	elif "small" in sam_checkpoint:
		model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
	elif "tiny" in sam_checkpoint:
		model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
	else:
		raise ValueError("Invalid SAM model checkpoint")
	sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint, device=device))
	return sam_predictor

		

def main():
	img_path = "example.png"
	image = Image.open(img_path).convert("RGB")
	# text_labels = [["cutting board", "sponge", "knife"],["cutting board", "sponge", "knife"]]
	text_labels = ["fruit", "hat", "coffee"]
	text_labels = ["fruit", "hat"]

	text_labels = [text_labels] * 2
	image = [image] * 2
	detector = OWLDetector(sam_checkpoint=SAM2_PATH, device="cuda:0")

	best_masks, best_scores = detector.mask_inference(image, text_labels)
	print(best_scores)
	print(np.shape(best_masks))
	mask_im = visualize_mask(np.array(image[0]), best_masks[0][1])
	cv2.imwrite("Mask.png", cv2.cvtColor(mask_im, cv2.COLOR_RGB2BGR))



	# results = detector.inference([image,image], text_labels)
	# print(results)
	# image = detector.generate_visualization(np.array(image), text_labels, results)
	# cv2.imwrite("Detection.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

	# cv2.imshow("Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	# cv2.waitKey(0)

	# mask = detector.get_obj_mask(image, "cutting board")
	# disp_im = visualize_mask(image, mask)
	# cv2.imshow("Mask", cv2.cvtColor(disp_im, cv2.COLOR_RGB2BGR))
	# cv2.waitKey(0)


if __name__ == "__main__":
	main()