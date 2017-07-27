import torch
import torchvision
import os
import numpy as np
from torch import optim
from torch.autograd import Variable
from model_InfoGAN import Discriminator
from model_InfoGAN import Generator


class Solver(object):
    def __init__(self, config, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.z_dim = config.z_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.build_model()
        
    def build_model(self):
        """Build generator and discriminator."""
        self.generator = Generator(z_dim=self.z_dim,
                                   image_size=self.image_size,
                                   conv_dim=self.g_conv_dim)
        self.discriminator = Discriminator(image_size=self.image_size,
                                           conv_dim=self.d_conv_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr, [self.beta1, self.beta2])

        self.D_criterion = torch.nn.BCELoss()
        self.G_criterion = torch.nn.BCELoss()
        
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
        
    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
    
    def reset_grad(self):
        """Zero the gradient buffers."""
        self.discriminator.zero_grad()
        self.generator.zero_grad()
    
    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # InfoGAN Function
    def gen_cont_c(self, n_size):
        return torch.Tensor(np.random.randn(n_size, 2) * 0.5 + 0.0)

    # InfoGAN Function
    def gen_disc_c(self, n_size):
        codes=[]
        code = np.zeros((n_size, 2))
        random_cate = np.random.randint(0, 2, n_size)
        code[range(n_size), random_cate] = 1
        codes.append(code)
        codes = np.concatenate(codes,1)
        return torch.Tensor(codes)

    def train(self):
        """Train generator and discriminator."""
        fixed_noise = self.to_variable(torch.randn(100, self.z_dim))
        total_step = len(self.data_loader)
        for epoch in range(self.num_epochs):
            for i, images in enumerate(self.data_loader):
                
                #===================== Train D =====================#
                images = self.to_variable(images)
                batch_size = images.size(0)
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                #InfoGAN
                cont_codes = self.to_variable(self.gen_cont_c(batch_size))
                disc_codes = self.to_variable(self.gen_disc_c(batch_size))

                # Train D to recognize real images as real.
                # outputs = self.discriminator(images)
                # real_loss = torch.mean((outputs - 1) ** 2)      # L2 loss instead of Binary cross entropy loss (this is optional for stable training)

                # Train D to recognize fake images as fake.
                infogan_inputs = torch.cat((noise, cont_codes, disc_codes), 1)
                fake_images = self.generator(infogan_inputs)
                #outputs = self.discriminator(fake_images)
                #fake_loss = torch.mean(outputs ** 2)

                new_inputs = torch.cat([images, fake_images])

                # Make Tmp Labels
                labels = np.zeros(2 * batch_size)
                labels[:batch_size] = 1
                labels = self.to_variable(torch.from_numpy(labels.astype(np.float32)))

                outputs = self.discriminator(new_inputs)

                # Backprop + optimize
                #d_loss = real_loss + fake_loss
                self.discriminator.zero_grad()

                Q_cx_dis = outputs[batch_size:, 3:5]
                codes = disc_codes[:, 0:2]
                condi_entropy = -torch.mean(torch.sum(Q_cx_dis * codes, 1))
                L_discrete = -condi_entropy

                Q_cx_cont = outputs[batch_size:, 1:3]
                L_conti = torch.mean(-(((Q_cx_cont - 0.0) / 0.5) ** 2))

                D_loss = self.D_criterion(outputs[:,0], labels) - 1.0 * L_discrete - 0.5 * L_conti

                D_loss.backward(retain_variables=True)
                self.d_optimizer.step()
                
                #===================== Train G =====================#
                """
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))
                
                # Train G so that D recognizes G(z) as real.
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images)
                g_loss = torch.mean((outputs - 1) ** 2)
                """
                # Backprop + optimize
                self.generator.zero_grad()

                # InfoGAN
                G_loss = self.G_criterion(outputs[batch_size:, 0], labels[:batch_size]) - 1.0 * L_discrete - 0.5 * L_conti

                G_loss.backward()
                self.g_optimizer.step()
    
                # print the log info
                if (i+1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], D_loss: %.4f, ' 
                          'G_loss: %.4f, G_loss: %.4f' 
                          %(epoch+1, self.num_epochs, i+1, total_step, 
                            D_loss.data[0], G_loss.data[0], G_loss.data[0]))

                # save the sampled images
                if (i+1) % self.sample_step == 0:
                    # -1 to 1 of First Continuous Code
                    tmp = np.zeros((100,2))
                    tmp[0:25,0] = np.linspace(-3,3,25)
                    tmp[25:50,1] = np.linspace(-3,3,25)
                    tmp[50:75,0] = np.linspace(-3,3,25)
                    tmp[75:100,1] = np.linspace(-3,3,25)
                    cont_codes = self.to_variable(torch.Tensor(tmp))

                    # 1 of First Categorical Code
                    tmp2 = np.zeros((100,2))
                    tmp2[0:50,0] = 1
                    tmp2[50:100,1] = 1
                    disc_codes = self.to_variable(torch.Tensor(tmp2))

                    # Fixed Noise = Zero
                    cont_noise = self.to_variable(torch.Tensor(np.zeros((100,self.z_dim))))

                    infogan_inputs = torch.cat((cont_noise, cont_codes, disc_codes), 1)

                    fake_images = self.generator(infogan_inputs)

                    torchvision.utils.save_image(self.denorm(fake_images.data), 
                        os.path.join(self.sample_path,
                                     'fake_samples-%d-%d.png' %(epoch+1, i+1)), nrow=10)
            
            # save the model parameters for each epoch
            g_path = os.path.join(self.model_path, 'generator-%d.pkl' %(epoch+1))
            d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' %(epoch+1))
            torch.save(self.generator.state_dict(), g_path)
            torch.save(self.discriminator.state_dict(), d_path)
            
    def sample(self):
        
        # Load trained parameters 
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' %(self.num_epochs))
        d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' %(self.num_epochs))
        self.generator.load_state_dict(torch.load(g_path))
        self.discriminator.load_state_dict(torch.load(d_path))
        self.generator.eval()
        self.discriminator.eval()
        
        # Sample the images
        noise = self.to_variable(torch.randn(self.sample_size, self.z_dim))
        fake_images = self.generator(noise)
        sample_path = os.path.join(self.sample_path, 'fake_samples-final.png')
        torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=12)
        
        print("Saved sampled images to '%s'" %sample_path)
