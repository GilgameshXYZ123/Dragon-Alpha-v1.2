/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu_fused;

import cifar10.resnet18_leaky_relu_fused.Net.ResNet18;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train3 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    static {
        eg.kaiming_fan_mode = Engine.FanMode.fan_in_out;
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
    
    static int batch_size = 64;//512;
    static float lr = 0.001f;//learning_rate
    
    public static void training(int nIter) {
        ResNet18 net = new ResNet18().train().init(eg).println(); //net.load();
        
        Optimizer opt = alpha.optim.Adam(net.params(), lr);
//        Optimizer opt = alpha.optim.SGD(net.params(), lr).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        eg.sync(false).check(false);
        SimpleTimer timer = new SimpleTimer().record();
        
        for(int i=0; i<nIter; i++) {
//            Tensor X = eg.zeros(batch_size, 128, 128, 3);
            Tensor X = eg.zeros(batch_size, 224, 224, 3);
            Tensor Y = eg.zeros(batch_size, 10);

            Tensor Yh = net.forward(X)[0];

            if (i % 10 == 0) {
                float ls = loss.loss(Yh, Y).get();
                System.out.println(i + ": loss = " + ls + ", lr = " + opt.learning_rate());
                System.gc();
            }

            net.backward(loss.gradient(Yh, Y));
            opt.update().clear_grads();
            net.gc();
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / nIter;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time / batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            //224*224*3: total = 35.466 -> 34.835 -> 33.603 ->  33.397 ->  33.179 -> 29.595 -> 28.954 -> 28.17 ->  27.709 -> 26.939
            //100 batch, 128*128*128*3: 22.486, 22.841 -> 19.215 -> 18.649 ->  18.471
            //25 epochs for Adam
            //50 epochs for SGD
            //30 epcohs for SGDMN
            training(100);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
