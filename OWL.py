from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib import pyplot as plt

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask = mask.astype(np.uint8)
	mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	if borders:
		contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
		# Try to smooth contours
		contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
		mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
	ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
	pos_points = coords[labels==1]
	neg_points = coords[labels==0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
	for i, (mask, score) in enumerate(zip(masks, scores)):
		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		show_mask(mask, plt.gca(), borders=borders)
		if point_coords is not None:
			assert input_labels is not None
			show_points(point_coords, input_labels, plt.gca())
		if box_coords is not None:
			# boxes
			show_box(box_coords, plt.gca())
		if len(scores) > 1:
			plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
		plt.axis('off')
		plt.show()

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
	detector = OWLDetector(sam_checkpoint="/home/mverghese/sam2/checkpoints/sam2.1_hiera_large.pt", device="cuda:1")

	results = detector.inference(image, text_labels)
	print(results)
	image = detector.generate_visualization(np.array(image), text_labels, results)

	cv2.imshow("Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	cv2.waitKey(0)

if __name__ == "__main__":
	main()