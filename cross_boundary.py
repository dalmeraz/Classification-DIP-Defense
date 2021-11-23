import ssim



def truncate(dip_trace, dip_trace_labels, t):
    a = dip_trace[t:]
    b = dip_trace_labels[t:]
    return a, b



def cross_boundary(dip_trace, dip_trace_labels, similar_threshold):
    cross_boundary_img = []
    cross_boundary_label = []
    for i in range(len(dip_trace)-1):
        img1 = dip_trace[i]
        img2 = dip_trace[i+1]
        similar = ssim.calculate_ssim(img1, img2)
        label_img1 = dip_trace_labels[i].index(max(dip_trace_labels[i]))
        label_img2 = dip_trace_labels[i+1],index(max(dip_trace_labels[i+1]))
        if (label_img1 != label_img2) and (similar > similar_threshold):
            cross_boundary_img.append(img1)
            cross_boundary_label.append(dip_trace_labels[i])

    return cross_boundary_img, cross_boundary_label


def create_boundary_img(img1, img2, img1_label, img2_label):
    mini = 1
    alpha_meta = 0
    for i in range(100):
        alpha = i / 100.0
        x_try = alpha * img1 + (1-alpha) * img2
        b = torch.round(x_try)
        diff = abs(victim_model.forward(b)[img1_label] - victim_model.forward(b)[img2_label])
        if diff < mini:
            mini = diff
            alpha_meta = alpha

    x_bd = torch.round(alpha_meta * img1 + (1-alpha_meta) * img2)
    return x_bd
        




def boundary_image(dip_trace, dip_trace_labels, similar_threshold, t):
    boundary_img = []
    dip_trace, dip_trace_labels = truncate(dip_trace, dip_trace_labels, t)
    cross, cross_label = cross_boundary(dip_trace, dip_trace_labels, similar_threshold)
    for i in range(len(cross)):
        img1 = cross[i]
        a = dip_trace.index(img1) + 1
        img2 = dip_trace[a]
        img1_label =cross_label[i].index(max(cross_label[i]))
        img2_label =dip_trace_labels[a].index(max(dip_trace_labels[a]))
        x_bd = create_boundary_img(img1, img2, img1_label, img2_label)
        boundary_img.append(x_bd)

    return boundary_img


def manifold_stitching(dip_trace, dip_trace_labels, similar_threshold, t, beta, origin_img):
    manifold_image_list = []
    boundary_image_list = boundary_image(dip_trace, dip_trace_labels, similar_threshold, t)
    K = len(boundary_image_list)
    for i in range(K):
        x_manifold = boundary_image_list[i] + beta * (boundary_image_list[i] - origin_img)
        manifold_image_list.append(x_manifold)

    x_rec = torch.zeros(manifold_image_list[0].size())

    for i in range(K):
        x_rec = x_rec + (manifold_image_list[i] / K)

    return x_rec
















