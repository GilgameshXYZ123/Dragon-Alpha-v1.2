/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.net;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * The backbone of VGG19 without stem and classifier.
 * @author Gilgamesh
 */
public class VGG19 extends Module {
    private static final long serialVersionUID = 1L;

    public Unit conv1 = nn.conv3D(false,  3, 64, 3, 1, 1);
    public Unit conv2 = nn.conv3D(false, 64, 64, 3, 1, 1);
    public Unit pool1 = nn.maxPool2D(2);//div = 2
    
    public Unit conv3 = nn.conv3D(false,  64, 128, 3, 1, 1);
    public Unit conv4 = nn.conv3D(false, 128, 128, 3, 1, 1);
    public Unit pool2 = nn.maxPool2D(2);//div = 4
    
    public Unit conv5 = nn.conv3D(false, 128, 256, 3, 1, 1);
    public Unit conv6 = nn.conv3D(false, 256, 256, 3, 1, 1);
    public Unit conv7 = nn.conv3D(false, 256, 256, 3, 1, 1);
    public Unit conv8 = nn.conv3D(false, 256, 256, 3, 1, 1);
    public Unit pool3 = nn.maxPool2D(2);//div = 8
   
    public Unit conv9 = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convA = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convB = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convC = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit pool4 = nn.maxPool2D(2);//div = 16
    
    public Unit convD = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convE = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convF = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit convG = nn.conv3D(false, 512, 512, 3, 1, 1);
    public Unit pool5 = nn.maxPool2D(2);//div = 32
    
    public Unit bn1, bn2, bn3, bn4, bn5, act;
    public VGG19(Unit activation, boolean batchNorm) {
        act = activation;
        bn1 = (batchNorm ? nn.fuse(nn.batchNorm( 64), act) : act);
        bn2 = (batchNorm ? nn.fuse(nn.batchNorm(128), act) : act);
        bn3 = (batchNorm ? nn.fuse(nn.batchNorm(256), act) : act);
        bn4 = (batchNorm ? nn.fuse(nn.batchNorm(512), act) : act);
        bn5 = (batchNorm ? nn.fuse(nn.batchNorm(512), act) : act);
    }
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = act.forward(conv1.forward(X));
        X = bn1.forward(conv2.forward(X));
        X = pool1.forward(X);
        
        X = act.forward(conv3.forward(X));
        X = bn2.forward(conv4.forward(X));
        X = pool2.forward(X);
          
        X = act.forward(conv5.forward(X));
        X = act.forward(conv6.forward(X));
        X = act.forward(conv7.forward(X));
        X = bn3.forward(conv8.forward(X));
        X = pool3.forward(X);
        
        X = act.forward(conv9.forward(X));
        X = act.forward(convA.forward(X));
        X = act.forward(convB.forward(X));
        X = bn4.forward(convC.forward(X));
        X = pool4.forward(X);
        
        X = act.forward(convD.forward(X));
        X = act.forward(convE.forward(X));
        X = act.forward(convF.forward(X));
        X = bn5.forward(convG.forward(X));
        X = pool5.forward(X);
        return X;
    }
}
