/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.vgg16;

import static imgnet.Config.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class speed 
{
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
    
    static int batch_size = 256;
    static float lr = 0.001f;//learning_rate
    
    public static void training(int nIter) {
        VGG16 net = new VGG16().train().init(eg).println();
        
        Optimizer opt = alpha.optim.Adam(net.params(), lr).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        eg.sync(false).check(false);
        SimpleTimer timer = new SimpleTimer().record();
        
        for (int i=0; i<nIter; i++) {
            Tensor X = eg.zeros(batch_size, 128, 128, 3);
            Tensor Y = eg.zeros(batch_size, 1000);

            Tensor Yh = net.forward(X)[0];

            if(i % 10 == 0)  {
                float ls = loss.loss(Yh, Y).get();
                System.out.println(i + ": loss = " + ls + ", lr = " + opt.learning_rate());
            }

            net.backward(loss.gradient(Yh, Y));
            opt.update().clear_grads();
            net.gc();
        }
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f * div / nIter;
        System.out.println("total = " + (1.0f * div / 1000));
        System.out.println("for each sample:" + time / batch_size);
    }
    
    public static void main(String[] args) {
        try { training(100); }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
