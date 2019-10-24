def show_image(img_ndarray, id):

    '''
    Visualize images resulted from calling vis_smpl_params in Jupyternotebook
    :param img_ndarray: Nx400x400x3
    '''

    import matplotlib as plt
    import os
    import numpy as np
    import cv2


    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    if not os.path.isdir("images"):
        os.makedirs("images")
    fig_name = "images/fig" + str(id) + ".png"
    plt.savefig(fig_name)
    plt.close()
    # fig.canvas.draw()
    # return True


