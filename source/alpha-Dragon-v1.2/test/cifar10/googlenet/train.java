/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.googlenet;

import cifar10.googlenet.Net.GoogLeNet;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
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
public class train 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.field_var_safe(false);
    }
    
    static int batch_size = 512;
    static float lr = 0.001f;
    
    public static void training(int epoch) {
        GoogLeNet net = new GoogLeNet().train().init(eg).println();
        
        Optimizer opt = alpha.optim.Adam(net.params(), lr).println();//30 epoch
//        Optimizer opt = alpha.optim.SGDMN(net.params(), lr).momentum(0.9f).nestorv(1).println();//30 epoch

        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        DataSet<byte[], Integer> train_set = Cifar10.train();
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        
        int batchIdx = 0; 
        eg.sync(false).check(false);
        SimpleTimer timer = SimpleTimer.clock();
        
        for(int i=0; i<epoch; i++) {
            System.out.println("\n\nepoch = " + i);
            for(iter.reset().shuffle_swap(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();
                Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
                Tensor Y = pair.label;
                
                Tensor Yh = net.forward(X)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(Yh, Y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }

                net.backward(loss.gradient(Yh, Y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f * div / 1000));
        System.out.println("for each sample:" + time / batch_size);
        net.save();
    }
    
    public static void main(String[] args) {
        try { training(5); }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
