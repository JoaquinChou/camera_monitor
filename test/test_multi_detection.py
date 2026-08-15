from models import Yolo26Detection
from tasks import MultiDetectionTask
from utils import read_img, save_img, plot_detect_results



def test_single_img(config_path, img_path, output_path, gpu_id):
    yolo26Detection = Yolo26Detection(config_path, gpu_id=gpu_id)
    multiDetectionTask = MultiDetectionTask(yolo26Detection)
    img = read_img(img_path)
    results = multiDetectionTask.run(img)
    print(results)
    plot_detect_results(results, img)
    save_img(img, output_path)

    print(f"Output image saved to {output_path}")



if __name__ == "__main__":
    config_path = "./configs/yolo26_onnx.json"
    img_path = "./data/images/test_frame_at_9m10s.jpg"
    output_path = "./output/images/test_frame_at_9m10s_res.jpg"
    gpu_id = 1
    test_single_img(config_path, img_path, output_path, gpu_id)