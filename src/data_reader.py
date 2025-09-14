import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

class DataReader:
    def __init__(self, viz_data=False):

        try:
            self.dataset_path = os.path.join("../dataset/anime_face_dataset/images")   # dataset directory
            # self.img = cv.imread(os.path.join(self.dataset_path, "0_2000.jpg"))

            if viz_data:
                self.visualize_data()
            

        except FileExistsError as e:
            print("Error:", e)

        except pd.errors.ParserError as e:
            print("Parsing error while reading CSV:", e)

        except Exception as e:
            print("Unexpected error:", e)

    def visualize_data(self):
        img = cv.imread(os.path.join(self.dataset_path, "0_2000.jpg"))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        print("Image shape:", img.shape)
        cv.imshow("Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        plt.imshow(img)
        plt.show()

# if __name__ == "__main__":
#     data_reader = DataReader(viz_data=False)
    # data_reader = DataReader(viz_data=True)