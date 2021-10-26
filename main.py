import matplotlib.pyplot as plt
import torch

import attack_model
import defense_model
import victim_model
import dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

log_every = 50
test = 'cifar10'
epochs = 3
learning_rate = 0.01

def show_img(img):
    img_np = img.detach().cpu().numpy()[0]
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1,2,0)
    plt.figure()
    plt.imshow(img_np, interpolation='lanczos')

def generate_dip_trace(attack, dip_trace_length, victim_model):
    dip_trace = []
    dip_trace_labels = []

    defender = defense_model.defense_model(attack)
    for i in range(dip_trace_length):
        dip_result = defender.forward()
        victim_result = victim_model.forward(dip_result)

        dip_trace.append(dip_result)
        dip_trace_labels.append(victim_result)
        
        if i % log_every == 0:
            print(i)
            show_img(dip_result)

    return dip_trace, dip_trace_labels




if __name__ == "__main__":
    loaders = dataset.dataset(test, 128)
    victim = victim_model.victim_model(test, learning_rate).train_model(epochs, loaders)
    attacker = attack_model.attack_model(victim)
    # attacks = attacker.attack_one_batch(loaders.test_loader)
    
    attacks = attacker.attack(loaders.test_loader)
    show_img(attacks[0]['attack'])
    show_img(attacks[0]['original'])
    print(len(attacks))
    generate_dip_trace(attacks[0]['attack'], 500, victim)
