/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.bernouli;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 * bernouli(sigmoid(X) > p, v1, v2).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Sigmoid_BernouliMul extends BernouliMul {
    private static final long serialVersionUID = 562781240330663L;
    
    public Sigmoid_BernouliMul(boolean inplace, float p, float v1, float v2) {
        super(inplace, p, v1, v2);
    }
    
    @Override
    protected InlineSigmoid_BernouliMul create_unit_core() { 
        return new InlineSigmoid_BernouliMul(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBernouliMul">
    public static class InlineSigmoid_BernouliMul extends InlineBernouliMul<Sigmoid_BernouliMul> {
        public InlineSigmoid_BernouliMul(Sigmoid_BernouliMul unit) { super(unit); }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) {
                X = eg.sigmoid(inplace, X);
                float exp = expect();//exp = (v1*p + v2*(1.0f - p))
                return (exp == 1.0f ? X : eg.linear(true, exp, X.c(), 0.0f));
            }
        
            Tensor[] outs = eg.sigmoid_bernouli_mul(X, ut.p, ut.v1, ut.v2);
            R = outs[1];
            return outs[0];//Y = outs[0]
        }
        //</editor-fold>
    }
    //</editor-fold>
}
