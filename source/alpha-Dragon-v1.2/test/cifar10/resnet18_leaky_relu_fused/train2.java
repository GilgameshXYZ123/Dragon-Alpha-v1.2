/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu_fused;

import cifar10.resnet18_leaky_relu_fused.Net.ResNet18;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.data.Transform;
import z.dragon.dataset.Cifar10;
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
public class train2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    static {
        eg.kaiming_fan_mode = Engine.FanMode.fan_in_out;
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
    
    static int batch_size = 512;//512;
    static float lr = 0.001f;//learning_rate
    
    static Transform<byte[][]> input_transform = (Engine eg, byte[][] value) -> {
        Tensor X = eg.img.pixels(value, Cifar10.picture_dim);
        X = eg.img.jit_color(0.3f, true, X, 0, 0, 0).c();
        return X.pixel_to_tensor(true).c();
    };
    
    public static void training(int epoch) {
        ResNet18 net = new ResNet18().init(eg).println(); 
        net.train();
        
        Optimizer opt = alpha.optim.AdamW(net.params(), lr, 1e-4f);
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        DataSet<byte[], Integer> train_set = Cifar10.train(input_transform);
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        eg.sync(false).check(false);
        
        int batchIdx = 0; 
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<epoch; i++) { 
            System.out.println("\n\nepoch = " + i);
            for(iter.reset().shuffle_swap(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();
                Tensor x = pair.input.ssub(true, 0.5f);
                Tensor y = pair.label;
                
                Tensor yh = net.forward(x)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(yh, y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(yh, y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();//3.7 - 11.187 = 7.48
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            //30.918
            //25 epochs for Adam
            //50 epochs for SGD
            //30 epcohs for SGDMN
            training(80);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
