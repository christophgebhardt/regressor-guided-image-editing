import torch

class ImageScorer:
    def __init__(self, pipe):
        self.pipe = pipe

    def score(self, image, prompts=None):
        prompts = prompts if prompts is not None else ["", ""]
        image = self.pipe.transform_image(image)
        score = self.pipe.guidance_classifier.predict_score(
            self.pipe.get_latents_from_img(image, torch.float16),
            self.pipe.pipe.scheduler.timesteps[-1],
            prompts)
        
        return score
    
    
    def rec_error(self, orig_image, adapted_image):
        '''
        Calculates reconstruction error
        '''
        orig_image_tensor = self.pipe.transform_image(orig_image)
        adapted_image_tensor = self.pipe.transform_image(adapted_image)
        return torch.mean(torch.abs(adapted_image_tensor - orig_image_tensor)).item()
    


    def print_score(self, score, label, orig_score=None):
        '''
        prints a given score. If an aditional "original score" if provided,
        the score and the diffrence to the original are printed.
        '''
        if orig_score is None:
            print(f"Score {label}: valence {score[0, 0].item():.4f}, arousal {score[0, 1].item():.4f}")
            return

        delta = score - orig_score
        print(f"Score {label}: valence {score[0, 0].item():.4f} delta {delta[0, 0].item():.4f}, "
            f"arousal {score[0, 1].item():.4f} delta {delta[0, 1].item():.4f}")
        return
    

