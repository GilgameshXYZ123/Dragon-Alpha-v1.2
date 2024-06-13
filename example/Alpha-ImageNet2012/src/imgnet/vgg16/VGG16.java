/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.vgg16;

import imgnet.Config;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
    });
 * @author Gilgamesh
 */
public class VGG16 extends Module
{
    Unit conv1 = nn.conv3D(false,  3, 64, 3, 1, 1);
    Unit conv2 = nn.conv3D(false, 64, 64, 3, 1, 1);
    Unit bn1 = nn.batchNorm_leakyRelu(nn.batchNorm(64), nn.leakyRelu());
    
    Unit conv3 = nn.conv3D(false,  64, 128, 3, 1, 1);
    Unit conv4 = nn.conv3D(false, 128, 128, 3, 1, 1);
    Unit bn2 = nn.batchNorm_leakyRelu(nn.batchNorm(128), nn.leakyRelu());
    
    Unit conv5 = nn.conv3D(false, 128, 256, 3, 1, 1);
    Unit conv6 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit conv7 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit bn3 = nn.batchNorm_leakyRelu(nn.batchNorm(256), nn.leakyRelu());
    
    Unit conv8 = nn.conv3D(false, 256, 512, 3, 1, 1);
    Unit conv9 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv10 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit bn4 = nn.batchNorm_leakyRelu(nn.batchNorm(512), nn.leakyRelu());
    
    Unit conv11 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv12 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv13 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit bn5 = nn.batchNorm_leakyRelu(nn.batchNorm(512), nn.leakyRelu());
    
    Unit classifier = nn.sequence(
            nn.fullconnect(false, 8192, 4096),
            nn.leakyRelu_dropout(nn.leakyRelu(), nn.dropout(0.9f)),
            nn.fullconnect(false, 4096, 2048),
            nn.leakyRelu_dropout(nn.leakyRelu(), nn.dropout(0.9f)),
            nn.fullconnect(true, 2048, 1000));
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = F.leakyRelu(conv1.forward(X));
        X = bn1.forward(conv2.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv3.forward(X));
        X = bn2.forward(conv4.forward(X)); 
        X = F.maxPool2D(2, X);      
          
        X = F.leakyRelu(conv5.forward(X)); 
        X = F.leakyRelu(conv6.forward(X));
        X = bn3.forward(conv7.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv8.forward(X));
        X = F.leakyRelu(conv9.forward(X));
        X = bn4.forward(conv10.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv11.forward(X));
        X = F.leakyRelu(conv12.forward(X));
        X = bn5.forward(conv13.forward(X));
        X = F.maxPool2D(2, X);
        
        X = classifier.forward(F.flatten(X));
        return X;
    }
    
    static String weight     =  Config.weight_home + "Alpha-v1.1-imgnet2012-VGG16.zip";
    static String opt_weight =  Config.weight_home + "Alpha-v1.1-imgnet2012-VGG16-opt.zip";
}
