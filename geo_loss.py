import torch
import torch.nn as nn
# Function to apply geometric ensemble augmentation to an input image.
# It takes a neural network (net), an image (img), and an optional flip direction (flip_dir)
def geometry_ensemble(net, img, flip_dir="H"):
    assert len(img.shape) == 4  # Ensure input image is in 4D (batch, channel, height, width)
    assert flip_dir in ["H", "V", None]  # Check that flip direction is valid
    # Determine which axes to flip based on the specified flip direction
    if flip_dir == "H":  # Horizontal flip
        flip_axes = [3]
    elif flip_dir == "V":  # Vertical flip
        flip_axes = [2]
    imgs = []
    # Rotate the image by 0, 90, 180, and 270 degrees
    for r in range(4):
        imgs.append(torch.rot90(img, r, [2, 3]))
    if flip_dir:  # If flipping is enabled
        flips = []
        for im in imgs:
            flips.append(torch.flip(im, flip_axes))
        imgs.extend(flips) # Add flipped images to the original rotated images
    outs = []
    # Process each augmented image through the network
    for r, im in enumerate(imgs):
        temp = net(im)
        if r < 4:  # Handle rotated images
            outs.append(torch.rot90(temp, -r, [2, 3]))  # Rotate back to original orientation
        else:  # Handle flipped images
            temp2 = torch.flip(temp, flip_axes)
            outs.append(torch.rot90(temp2, -(r%4), [2, 3]))  # Rotate back to original orientation
    # Sum the results of all processed images
    for i in range(1, len(outs)):
        outs[0] += outs[i]
    # Return the averaged output
    return outs[0] / len(outs)


if __name__ == "__main__":
    random_seed = 2020
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    x = torch.cat((x, x, x, x), dim=0)
    print(x)
    crit = nn.L1Loss()
    #net = nn.Identity()
    net = nn.Conv2d(1, 1, 3, 1, 1)
    net.weight.data.normal_()
    net.bias.data.fill_(0)
    import torch.optim as optim
    sgd = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for i in range(10000):
        sgd.zero_grad()
        y = net(x)
        out = geometry_ensemble(net, x, flip_dir="H")
        loss = crit(out, y)
        loss.backward()
        sgd.step()
        print(f"{i}: {loss.item():.8f}")
    print(out)
