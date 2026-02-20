import os

from adapt_images.scoring import ImageScorer

class OutputImageManager:
    """
    Class managing results of pipeline
    """

    def __init__(self, scorer: ImageScorer, output_path = "."):
        self.output_path = output_path
        self.scorer = scorer

        # set by the ImgaeAdapter for the upcomming image
        self.image_name: str = None
        self.orig_image_score = None
        self.orig_image = None


    def callback(self, adapted_image, label=None):
        '''
        Called by the pipeline after adapting an image
        '''
        image_path = f"{self.output_path}/{label}/{self.image_name}.jpg"
        if not os.path.exists(f"{self.output_path}/{label}"):
            os.makedirs(f"{self.output_path}/{label}")

        adapted_image.save(image_path)

        # score new image
        score = self.scorer.score(adapted_image)
        self.scorer.print_score(score, "adapted", self.orig_image_score)

        # calculate and print reconstruction error
        rec_error = self.scorer.rec_error(self.orig_image, adapted_image)
        print("Reconstruction error: {:.4f}".format(rec_error))




    def set_image_name(self, name: str):
        self.image_name = name

    def set_orig_image_score(self, score):
        self.orig_image_score = score

    def set_orig_image(self, img):
        self.orig_image = img

