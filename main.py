import matplotlib.pyplot as plt
import torch
import numpy as np

import attack_model
import defense_model
import victim_model
import dataset

import cross_boundary
import config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def generate_dip_trace(attack, dip_trace_length, victim_model):
    dip_trace = []
    dip_trace_labels = []

    defender = defense_model.defense_model(attack)

    for i in range(dip_trace_length):
        dip_result = defender.forward()
        victim_result = victim_model.forward(dip_result)

        dip_trace.append(dip_result)
        dip_trace_labels.append(victim_result)

    return dip_trace, dip_trace_labels

def generate_label(label):
    label_np = label.detach().cpu().numpy()
    return config.dataset_labels[label_np.argmax()]

def generate_image(img):
    img_np = img.detach().cpu().numpy()
    return img_np.transpose(1,2,0)

    
def visualize_dip_trace(dip_trace, dip_trace_labels, show_every):
    c = 6
    # TODO: Implement line below better, using 0.499 is hacky
    r = round((len(dip_trace)//show_every)/ c + 0.499)
    
    fig2 = plt.figure(figsize=(r*2, c*2))
    fig2.suptitle("DIP Trace")
    
    for i in range (0,len(dip_trace)//show_every):
        fig2.add_subplot(r,c,i+1)
        plt.title(generate_label(dip_trace_labels[i*show_every][0]))
        plt.axis('off')
        plt.imshow(generate_image(dip_trace[i*show_every][0]))
    plt.show()
    

def visualize_attacks(attacks):
    c = 2
    r = len(attacks)
    
    fig = plt.figure(figsize=(r*2, c*2))
    fig.suptitle("Attacks")
    
    for i in range (len(attacks)):
        fig.add_subplot(r,c, i*2+1)
        attack_img = generate_image(attacks[i]['attack'])
        attack_label = config.dataset_labels[attacks[i]['attack label']]
        plt.title("Attack "+ str(i)+": "+ attack_label)
        plt.axis('off')
        plt.imshow(attack_img)
        
        fig.add_subplot(r,c, i*2+2)
        original_label = config.dataset_labels[attacks[i]['original label']]
        original_img = generate_image(attacks[i]['original'])
        plt.title("Original "+ str(i)+": "+ original_label)
        plt.axis('off')
        plt.imshow(original_img)
    plt.show()

        
if __name__ == "__main__":
    loaders = dataset.dataset(config.test, 4)
    
    if config.pretrained:
        print('Loading pretrained victim...')
        victim = torch.load(config.pretrained_path)
    else:
        print("Training victim model...")
        victim = victim_model.victim_model(config.test, config.learning_rate).train_model(config.epochs, loaders)
    
    
    successful_attack = False
    attack_eps = 0.1
    while not successful_attack:
        attacks = attack_model.attack_model(victim, attack_eps).attack(loaders.test_loader)
        
        if len(attacks) > 0:
            print("Attack successful")
            successful_attack = True
            visualize_attacks(attacks)
        else:
            attack_eps += 0.05
            print("No successful attack, increasing eps to ", str(attack_eps))

    # Attack index
    ai = 0
    print("Generating dip trace...")
    dip_trace, dip_trace_labels = generate_dip_trace(attacks[ai]['attack'], config.dip_trace_length, victim)
    visualize_dip_trace(dip_trace, dip_trace_labels, config.log_every)
    
    x_rec = cross_boundary.manifold_stitching(dip_trace, dip_trace_labels, config.similar_threshold, config.t, config.beta, attacks[ai]['original'], victim)
    
    
    
    
