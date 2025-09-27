import argparse
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Read a ROS bag and print its contents.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag_file, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topics = reader.get_all_topics_and_types()
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        
        for topic in topics:
            if topic_name==topic.name:
                msg_type = get_message(topic.type)
                msg = deserialize_message(data, msg_type)
                if topic_name=="/boson_cam/image_raw":
                    # Convert ROS image message to OpenCV format
                    # Assuming it's a sensor_msgs/Image message
                    if hasattr(msg, 'data') and hasattr(msg, 'height') and hasattr(msg, 'width'):
                        # Convert the image data to numpy array
                        img_data = np.frombuffer(msg.data, dtype=np.uint8)
                        # Reshape to image dimensions
                        img = img_data.reshape((msg.height, msg.width, -1))
                        
                        # Display the image
                        cv2.imshow("ROS Image", img)
                        
                        # Wait for key press (0 means wait indefinitely)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):  # Press 'q' to quit
                            cv2.destroyAllWindows()
                            break
                    continue
                else:
                    # Do something with other message types here
                    print(msg)
    
    # Clean up OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()