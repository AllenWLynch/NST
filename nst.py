import tensorflow as tf
import numpy as np
import PIL.Image as img
import argparse
import os.path

def prep_VGG19():
    
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    
    vgg.trainable = False
    
    features_list = [tf.reshape(tf.transpose(layer.output, perm = [0,3,1,2]), [1,layer.output.shape[-1],-1])
                 for layer in vgg.layers]
    
    extractor = tf.keras.Model(inputs = vgg.input, outputs = features_list)
    
    return extractor


def content_loss(Fp, Fc):
    
    return 0.5 * tf.reduce_sum(tf.square(Fp - Fc))


def gram(F):
    
    return tf.matmul(F, F, transpose_b = True)

def layer_gram_loss(Fp, Fc):

    (_, nl, ml) = Fp.shape
    
    return tf.reduce_sum(tf.square(gram(Fp) - gram(Fc)))/(4 * nl**2 * ml**2)    



def gram_loss(W, features_product, features_style):
    
    return tf.reduce_sum([
        (1/len(W)) * layer_gram_loss(features_product[layernum], features_style[layernum])
        for layernum in W
    ])


def train_step(grad_image, 
              feature_model, 
              optimizer, 
              content_features, 
              style_features, 
              content_layer,
              style_layers, 
              content_weight,
              style_weight,
              tvl_weight):
    
    assert(type(content_layer) == int)
    assert(type(style_layers) == list)
    
    with tf.GradientTape() as tape:
        
        image_features = feature_model(preprocess_image(grad_image))
        
        loss = content_weight * content_loss(image_features[content_layer], content_features[content_layer]) + style_weight * gram_loss(style_layers, image_features, style_features)
        loss += tvl_weight * tf.image.total_variation(grad_image)
    
    grads = tape.gradient(loss, grad_image)
    optimizer.apply_gradients([(grads, grad_image)])
    grad_image.assign(tf.clip_by_value(grad_image, 0., 1.))
    
    return loss


def get_image(path):
    image = img.open(path)
    image = image.resize((224,224))
    x = np.array(image)
    x = np.expand_dims(x, 0)
    x = np.ndarray.astype(x, dtype = np.float32)
    return x / 255.


def preprocess_image(image):
    image = image * 255
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

def extant_file(x):
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def output_name(x):
    dirname, filename = os.path.dirname(x), os.path.basename(x)
    if not dirname == '' and not os.path.exists(dirname):
        raise argparse.ArgumentTypeError("{} is not a valid directory".format(dirname))
    if '.' in filename:
        filename = filename.split('.')[0]
    return os.path.join(dirname, filename)
        
parser = argparse.ArgumentParser()
parser.add_argument('content_path', type=extant_file, help='Path to content image')
parser.add_argument('style_path', type=extant_file, help='Path to style image')
parser.add_argument('output_path', type=output_name,help='Filename for stylized image, not including file extension.')
parser.add_argument('--start_from_image', metavar='FILE', type=extant_file,help='Start from a particular image, useful for continuing to optimize on an image')
parser.add_argument('--save_gif',help='Save a gif of evolution of image to same name as output.', action='store_true')
parser.add_argument('-e','--epochs', type=int, help='Number of epochs to optimize for.',default = 200)
parser.add_argument('-s', '--steps_per_epoch',type=int,help='Number of iterations to run in each epoch.',default = 25)
parser.add_argument('--start_from_content', help='Start optimizing from the content image, speeds up stylizing.', action = 'store_true')
parser.add_argument('--content_weight', type=float,default=1e4)
parser.add_argument('--style_weight',type=float,default=1e-2)
parser.add_argument('--total_variation_loss_weight',type=float,default=30)
parser.add_argument('-lr','--learning_rate',type=float,default=0.02,help='Learning rate for Adam optimizer.')

def NST(content_image, 
        style_image, 
        initial_image,
        epochs = 200,
        steps_per_epoch = 5,
        content_layer = 18, 
        style_layers = [1,4,7,12,17], 
        content_weight = 1e4, 
        style_weight = 1e-2,
        learning_rate = 0.02,
        total_variation_loss_weight = 30,       
       ):
    
    extractor = prep_VGG19()

    grad_image = tf.Variable(initial_image)

    content_features, style_features = extractor(preprocess_image(content_image)), extractor(preprocess_image(style_image))
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
        
    gif = np.zeros((epochs,224,224,3))
    
    try:
    
        for epoch in range(epochs):
            for step in range(steps_per_epoch):

                loss = train_step(grad_image, 
                  extractor, 
                  optimizer, 
                  content_features, 
                  style_features, 
                  content_layer,
                  style_layers, 
                  content_weight,
                  style_weight,
                  total_variation_loss_weight)

            print('Epoch: {}, Loss: {}'.format(str(epoch), str(loss.numpy()[0])))
            
            gen_image = np.squeeze(grad_image.numpy())
            
            gif[epoch] = gen_image
    except KeyboardInterrupt:
        pass
    
    return gif, gif[epoch - 1]


if __name__ == "__main__": 

    args = parser.parse_args()

    if args.content_weight:
        assert(args.content_weight > 0),'Content weight must be greater than 0'
    if args.style_weight:
        assert(args.style_weight > 0), 'Style weight must be greater than 0'
    if args.epochs:
        assert(args.epochs > 0), 'Num epochs must be greater than 0'
    if args.steps_per_epoch:
        assert(args.steps_per_epoch > 0), 'Steps per epoch must be greater than 0'
    if args.total_variation_loss_weight:
        assert(args.total_variation_loss_weight > 0 ), 'TVL must be greater than 0'
    assert(not (args.start_from_content and args.start_from_image)), 'Cannot start from both content and other specified image, choose one'

    content_image, style_image = get_image(args.content_path), get_image(args.style_path)

    if args.start_from_content:
        optimize_image = np.copy(content_image)
    elif args.start_from_image:
        optimize_image = get_image(args.start_from_image)
    else:
        optimize_image = np.ndarray.astype(np.random.rand(1,224,224,3), np.float32)
    

    gif_array, final_image = NST(content_image, 
                                style_image, 
                                optimize_image,
                                args.epochs,
                                steps_per_epoch = args.steps_per_epoch,
                                content_weight = args.content_weight, 
                                style_weight = args.style_weight,
                                learning_rate = args.learning_rate,
                                total_variation_loss_weight = args.total_variation_loss_weight,       
                            )

    print('Optimization complete!')
    print('Saving file: ' + args.output_path + '.jpg')
    
    final_image = img.fromarray((final_image * 255).astype('uint8'), 'RGB')
    final_image.save(args.output_path + '.jpg')

    if args.save_gif:
        print('Saving file: ' + args.output_path + '.gif')
        valid_images = [
            img.fromarray((im * 255).astype('uint8'), 'RGB')
            for im in gif_array if np.sum(im) > 0.
        ]

        valid_images[0].save(args.output_path + '.gif', save_all=True, 
                    append_images = valid_images[1:] + [valid_images[-1] for i in range(len(valid_images)//4)],
                    optimize=False, duration=50, loop=0)





