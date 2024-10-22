/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg16_5x5;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class VGG16_5x5 extends Module
{
    Unit conv1 = nn.conv3D(false,  3, 64, 5, 1, 2);
    Unit conv2 = nn.conv3D(false, 64, 64, 5, 1, 2);
    Unit bn1 = nn.batchNorm_leakyRelu(
            nn.batchNorm(64),
            nn.leakyRelu());
    
    Unit conv3 = nn.conv3D(false,  64, 128, 5, 1, 2);
    Unit conv4 = nn.conv3D(false, 128, 128, 5, 1, 2);
    Unit bn2 = nn.batchNorm_leakyRelu(
            nn.batchNorm(128),
            nn.leakyRelu());
    
    Unit conv5 = nn.conv3D(false, 128, 256, 5, 1, 2);
    Unit conv6 = nn.conv3D(false, 256, 256, 5, 1, 2);
    Unit conv7 = nn.conv3D(false, 256, 256, 5, 1, 2);
    Unit bn3 = nn.batchNorm_leakyRelu(
            nn.batchNorm(256),
            nn.leakyRelu());
    
    Unit conv8 = nn.conv3D(false, 256, 512, 5, 1, 2);
    Unit conv9 = nn.conv3D(false, 512, 512, 5, 1, 2);
    Unit conva = nn.conv3D(false, 512, 512, 5, 1, 2);
    Unit bn4 = nn.batchNorm_leakyRelu(
            nn.batchNorm(512),
            nn.leakyRelu());
    
    Unit convb = nn.conv3D(false, 512, 512, 5, 1, 2);
    Unit convc = nn.conv3D(false, 512, 512, 5, 1, 2);
    Unit convd = nn.conv3D(false, 512, 512, 5, 1, 2);
    Unit bn5 = nn.batchNorm_leakyRelu(
            nn.batchNorm(512),
            nn.leakyRelu());
    
    Unit classifier = nn.sequence(
            nn.fullconnect(false, 512, 256),
            nn.leakyRelu_dropout(
                    nn.leakyRelu(), 
                    nn.dropout(0.9f)),
            nn.fullconnect(false, 256, 128),
            nn.leakyRelu_dropout(
                    nn.leakyRelu(), 
                    nn.dropout(0.9f)),
            nn.fullconnect(true, 128, 10));
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = F.leakyRelu(conv1.forward(X));//div2: channel = 3 -> 64
        X = bn1.forward(conv2.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv3.forward(X));//div4: channel 64 -> 128
        X = bn2.forward(conv4.forward(X));
        X = F.maxPool2D(2, X);
          
        X = F.leakyRelu(conv5.forward(X));//div8: channel 128 -> 256
        X = F.leakyRelu(conv6.forward(X));
        X = bn3.forward(conv7.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv8.forward(X));//div16: 256 -> 512
        X = F.leakyRelu(conv9.forward(X));
        X = bn4.forward(conva.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(convb.forward(X));//div32: 512 -> 512
        X = F.leakyRelu(convc.forward(X));
        X = bn5.forward(convd.forward(X));
        X = F.maxPool2D(2, X);
        
        X = classifier.forward(F.flatten(X));
        return X;
    }
    
    static String weight = "C:\\Users\\Gilgamesh\\Desktop\\Alpha-v1.1-VGG16_5x5-cifar10.zip";
    public void load() { alpha.stat.load_zip(this, weight); };
    public void save() { alpha.stat.save_zip(this, weight); }
}
