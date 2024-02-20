from DiffuGAN.GAN.GAN import arg_parse, main
from .wgan import WGAN

if __name__ == "__main__":
    args = arg_parse()
    main(args=args, gan_object=WGAN)
