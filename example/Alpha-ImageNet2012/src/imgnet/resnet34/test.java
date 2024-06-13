/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.resnet34;

import static imgnet.Config.ImgNet2012_trainset;
import static imgnet.Config.ImgNet2012_valset;
import static imgnet.Config.eg;
import imgnet.resnet34.Net.ResNet34;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.Pair;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static int batch_size = 256;
    
    public static void test() {
        ResNet34 net = new ResNet34().eval().init(eg).println();
        alpha.stat.load_zip(net, ResNet34.weight);
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        BufferedTensorIter iter = ImgNet2012_trainset.init().shuffle_sort().buffered_iter(eg, batch_size);
        
        int batchIdx = 0;
        double accuracy = 0; 
        eg.sync(false).check(false);    
        
        for(; iter.hasNext(); batchIdx++) {
            Pair<Tensor, Tensor> pair = iter.next();
            Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
            Tensor Y = pair.label;
            
            Tensor Yh = net.forward(X)[0];
            
            System.out.println(batchIdx + ", loss = " + loss.loss(Yh, Y).get());
            
            Tensor predict = eg.row_max_index(Yh).c();//<int 32>
            Tensor label = eg.row_max_index(pair.label).c();//<int 32>
            
//            Vector.println("predict: ", predict.value_int32());
//            Vector.println("label: ", label.value_int32());
            
            float eq = eg.straight_equal(predict.c(), label.c()).get();
            System.out.println(batchIdx + ", eq = " + eq);
            accuracy += eq;
            
            net.gc(); predict.delete(); label.delete(); pair.label.delete();
        }
        
        System.out.println("accuracy = " + accuracy / batchIdx);
    }
    
    public static void main(String[] args) {
        try { test(); }
        catch(Exception e) { e.printStackTrace(); }
    }
}
