import torchattacks
import torch

class attack_model():

    def __init__(self, victim_model):
        self.attack_model = torchattacks.PGD(victim_model, eps=8/255, alpha=2/255, steps=4)
        # self.attack_model = torchattacks.PGD(victim_model, eps=0.8, alpha=2/255, steps=4)
        self.victim_model = victim_model

    def attack(self, test_loader):
        images, labels = next(iter(test_loader))
        for  images, labels in test_loader:
            original_outputs = self.victim_model(images)
            _, original_predictions = torch.max(original_outputs.data, 1) 
    
            attack_images = self.attack_model(images, labels)#.cuda()
            attack_outputs = self.victim_model(attack_images)
            _, attack_predictions = torch.max(attack_outputs.data, 1) 
    
            successful_attacks = []
            #print(labels.shape[0])
            for i in range(labels.shape[0]):
                if labels[i] == original_predictions[i] and labels[i] != attack_predictions[i]:
                    successful_attacks.append({'attack':attack_images[i], 'original': images[i], 'label': labels[i]})
        return successful_attacks

    def attack_one_batch(self, test_loader):
        images, labels = next(iter(test_loader))

        original_outputs = self.victim_model(images)
        _, original_predictions = torch.max(original_outputs.data, 1) 

        attack_images = self.attack_model(images, labels)#.cuda()
        attack_outputs = self.victim_model(attack_images)
        _, attack_predictions = torch.max(attack_outputs.data, 1) 

        successful_attacks = []
        for i in range(labels.shape[0]):
            print(labels[i], original_predictions[i], attack_predictions[i])
            if labels[i] == original_predictions[i] and labels[i] != attack_predictions[i]:
                successful_attacks.append({'attack':attack_images[i], 'original': images[i], 'label': labels[i]})
        return successful_attacks




