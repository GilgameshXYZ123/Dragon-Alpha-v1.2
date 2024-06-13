/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool2d;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreAdaptiveMaxPool2D <T extends SimpleUnit> extends CoreAdaptivePool2D<T>
{
    protected boolean training;

    transient protected Tensor Index;//Tensor<int32>

    public CoreAdaptiveMaxPool2D(T unit, boolean training,
            int out_height, int out_width) {
        super(unit, out_height, out_width);
        this.training = training;
    }

    public final boolean training() { return training; } 
    public Tensor Index() { return Index; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void variables(Tensor.TensorSet set) {
        super.variables(set);
        set.add(Index);
    }

    @Override
    public void gc() {
        super.gc(); 
        if(Index != null) { Index.delete(); Index = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        __adaptive__(X);
        if(!training) return eg.pool2D_max(X, FH, FW, OH, OW, 
                sh, sw, 0, 0);

        Tensor[] result = eg.pool2D_max_indexed(X, FH, FW, OH, OW, 
                sh, sw, 0, 0);//training: compute Index for: X -> Y
        Index = result[1];
        return result[0];//deltaY = result[0]
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.unpool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, 
                sh, sw, 0, 0);
        return (grad_inplace? 
                deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                deltaX.dual(()-> { Index.delete(); }));
    }
    //</editor-fold>
}
