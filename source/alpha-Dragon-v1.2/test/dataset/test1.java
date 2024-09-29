/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataset;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.DataSet;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Engine;

/**
 *
 * @author Gilgamesh
 */
public class test1 {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1(alpha.MEM_1GB * 4));
    
    public static void main(String[] args) {
        DataSet<byte[], Integer> dataset = Cifar10.train();
        System.out.println(dataset);
        
        DataSet<byte[], Integer>[] subsets = dataset.class_split(0.9f);
        DataSet<byte[], Integer> first = subsets[0];
        DataSet<byte[], Integer> last  = subsets[1];
        
        System.out.println(first);
        System.out.println(first.class_sample_num());
        System.out.println(last);
        System.out.println(last.class_sample_num());
    }
}
