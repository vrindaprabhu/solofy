import logging
import ray

from src.bbox_detect import yolov3_main
from src.saliency_detect import u2net_main
from src.inpaint import WGAN_tester
from src.create_mask import create_mask_bgreplace, create_mask_inpaint
from utils.rescale_image import rescale_image
from utils.logger import Logger
from config.solofy import *


if __name__ == "__main__":
    ray.init()
    LOG_LEVEL = logging.INFO

    logger = Logger.get_instance(LOG_LEVEL).get_logger()
    logger.info("=== Starting a new solofy activity. ===")

    logger.info("Calling the detection and saliency functions")
    ret_func1 = yolov3_main.remote()
    ret_func2 = u2net_main.remote()

    ret1, ret2 = ray.get([ret_func1, ret_func2])

    bboxes, img_shape = ret1
    original_h, original_w, _ = img_shape

    logger.info("Creating mask for inpainting")
    create_mask_inpaint(list(bboxes.values()), img_shape)

    scaled_boxes, resized_detection = rescale_image(
        bboxes,
        original_h,
        original_w,
    )

    # cv2.imshow(mat=resized_detection, winname="DetectionWindow")
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    logger.info("Obtaining the user choice for the PersonOfInterest")
    choice_dict = dict(zip(range(len(bboxes)), bboxes.keys()))
    print(choice_dict)
    choice = int(input("Enter which person you want in the photograph: "))
    print(f"You selected: {choice_dict[choice]}")
    logger.info(f"The bounding box choice selected was {choice_dict[choice]}")

    print("")

    logger.info("Inpainting started for the current activity")
    ray.get(WGAN_tester.remote())

    logger.info("Preparing final output image")
    create_mask_bgreplace(scaled_boxes[choice])

    logger.info("=== Final image created. Activity is done. ===")
