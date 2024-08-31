/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paper.net_present;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.simple.math2.LeakyRelu;
import z.dragon.nn.unit.simple.math2.Relu;

/**
 *
 * @author Gilgamesh
 */
public class GSter 
{
    boolean biased;
    boolean inplace;
  
    Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    public void main(String[] args) {
        Tensor X = null;
        
        Relu relu = nn.relu(inplace); 
        
        relu.forward(X);  
        
        
        eg.relu(inplace, X);
        
        
        float k;
        
        LeakyRelu lr = nn.leakyRelu(true, 0.01f);
        
        

        lr.setNegativeSlope(0.02f);//setter
        k = lr.getNegativeSlope();//getter
        System.out.println(k);
        k = lr.negative_slop(0.02f).negative_slope();//alpha.format
        System.out.println(k);
        
        
       
        
        
        
        k *= 1;
        System.out.println(k);
        
        
        
        
        
        
    }
    
    
}
