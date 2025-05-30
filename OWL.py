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
	def __init__(self, model_name="google/owlv2-base-patch16-ensemble", sam_checkpoint=None, device=None):
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
			self.sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint))
		else:
			self.sam_predictor = None

			

		self.colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
	
	def inference(self, image, text_labels):
		if isinstance(image, np.ndarray):
			image = Image.fromarray(image)
		inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
		outputs = self.model(**inputs)
		target_sizes = torch.tensor([(image.height, image.width)])
		results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
		return results

	def generate_visualization(self, image, labels, results):
		for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
			box = [round(i, 2) for i in box.tolist()]
			color = self.colors[label % len(self.colors)]
			image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
			image = cv2.putText(image, f"{labels[label]}: {round(score.item(), 3)}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return image

	def get_obj_box(self, image, obj_label):
		results = self.inference(image, [obj_label])
		scores = results[0]["scores"].cpu().detach()
		
		if len(scores) == 0:
			return np.zeros(4)
		box = results[0]["boxes"][np.argmax(scores)].tolist()
		box = [int(i) for i in box]
		return box
	
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
		

		

		

def main():
	img_path = "Test_Image.png"
	image = Image.open(img_path).convert("RGB")
	text_labels = ["cutting board", "sponge", "knife"]
	detector = OWLDetector(sam_checkpoint=SAM2_PATH, device="cuda:0")

	results = detector.inference(image, text_labels)
	print(results)
	image = detector.generate_visualization(np.array(image), text_labels, results)

	cv2.imshow("Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	cv2.waitKey(0)

	mask = detector.get_obj_mask(image, "cutting board")
	disp_im = visualize_mask(image, mask)
	cv2.imshow("Mask", cv2.cvtColor(disp_im, cv2.COLOR_RGB2BGR))
	cv2.waitKey(0)


if __name__ == "__main__":
	main()