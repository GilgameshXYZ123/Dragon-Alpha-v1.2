/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.googlenet;

import static cifar10.Config.eg;
import cifar10.googlenet.Net.GoogLeNet;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.data.TensorIter;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static int batch_size = 128;
    
    public static void test()  {
        GoogLeNet net = new  GoogLeNet().eval().init(eg).println(); 
        net.load(); 
      
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
//        DataSet<byte[], Integer> dataset = Cifar10.train();
        DataSet<byte[], Integer> dataset = Cifar10.test();
        
        int batchIdx = 0;
        double accuracy = 0;
        eg.sync(false).check(false);
        
        for(TensorIter iter = dataset.batch_iter(); iter.hasNext(); batchIdx++) {
            Pair<Tensor, Tensor> pair = iter.next(eg, batch_size);
            Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
            Tensor Y = pair.label;
            
            Tensor Yh = net.forward(X)[0];
            
            System.out.println("loss = " +  loss.loss(Yh, Y).get());
            
            Tensor predict = eg.row_max_index(Yh);
            Tensor label = eg.row_max_index(Y);
            
            Vector.println("predict: ", predict.value_int32());
            Vector.println("label: ", label.value_int32());
            
            float eq = eg.straight_equal(predict, label).get();
            System.out.println("eq = " + eq);
            accuracy += eq;
            
            net.gc(); predict.delete(); label.delete(); 
        }
        
        System.out.println("accuracy = " + accuracy / batchIdx);
    }
    
    public static void main(String[] args) {
        try { test(); }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
