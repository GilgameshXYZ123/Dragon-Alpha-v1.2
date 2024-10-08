/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.dropout;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 * Relu(dropout(X))
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Relu_Dropout extends Dropout {
    private static final long serialVersionUID = 562781240350001L;
    
    public Relu_Dropout(boolean inplace, float nonzero_p) {
        super(inplace, nonzero_p);
    }
    
    @Override
    protected InlineRelu_Dropout create_unit_core() {
        return new InlineRelu_Dropout(this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineRelu_Dropout">
    public static class InlineRelu_Dropout extends InlineDropout<Relu_Dropout> {
        public InlineRelu_Dropout(Relu_Dropout unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) return eg.relu(inplace, X);//exp = (1/p)*p + 0*(1-p) = 1

            Tensor[] outs = eg.relu_dropout(X,ut.p);
            R = outs[1];
            return outs[0];
        }
        //</editor-fold>
    }
    //</editor-fold>
}
