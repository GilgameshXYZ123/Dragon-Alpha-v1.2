/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.dual;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.Trace;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.dual.DualUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class DualCore<T extends DualUnit> extends UnitCore<T> {
    //<editor-fold defaultstate="collapsed" desc="member-parameters">
    transient private final UnitCoreMap<Object> arc = new UnitCoreMap<>();//solve the topology
    
    transient private Tensor X1, deltaX1;//input, input.gradients
    transient private Tensor X2, deltaX2;
    transient private Tensor Y, deltaY;//output, output.gradient
    transient private boolean last_need_grads1;//mapping: X1
    transient private boolean last_need_grads2;//mapping: X2
    
    transient private int X1_mod_count = -1;
    transient private int X2_mod_count = -1;
    transient private int Y_mod_count  = -1;
    //</editor-fold>
    
    public DualCore(T unit) { super(unit); }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final Tensor X1() { return X1; }
    public final Tensor X2() { return X2; }
    public final Tensor Y() { return Y; }
    
    public final Tensor deltaX1() { return deltaX1; }
    public final Tensor deltaX2() { return deltaX2; }
    public final Tensor deltaY() { return deltaY; }
    
    protected final Tensor holdX1() { return (X1.check() ? X1.hold(X1_mod_count, name() + ".X1") : X1); } 
    protected final Tensor holdX2() { return (X2.check() ? X2.hold(X2_mod_count, name() + ".X2") : X2); } 
    protected final Tensor holdY() { return (Y.check() ? Y.hold(Y_mod_count, name() + ".Y") : Y); }
    
    protected final boolean is_holdX1() { return X1.is_hold(X1_mod_count); }
    protected final boolean is_holdX2() { return X2.is_hold(X2_mod_count); }
    protected final boolean is_holdY() { return Y.is_hold(Y_mod_count); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public Collection<UnitCore<?>> next() { return arc.keySet(); }

    @Override 
    public void variables(TensorSet set) { 
        set.add(X1, X2, Y, deltaX1, deltaX2, deltaY); 
    }
    
    @Override
    public void gc() {
        if(X1 != null) { X1.delete(); X1 = null; }
        if(X2 != null) { X2.delete(); X2 = null; }
        if(Y != null) { Y.delete(); Y = null; }
        if(deltaX1 != null) { deltaX1.delete(); deltaX1 = null; }
        if(deltaX2 != null) { deltaX2.delete(); deltaX2 = null; }
        if(deltaY != null) { deltaY.delete(); deltaY = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor __forward__(Engine eg, Tensor X1, Tensor X2);
    
    @Override
    public Tensor[] forward(Tensor... input) {
        arc.clear();//all one-off UnitCores reference has been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X1 = input[0]; Trace trace1 = X1.trace(); if(trace1 != null) trace1.callback(this, 0);
        X2 = input[1]; Trace trace2 = X2.trace(); if(trace2 != null) trace2.callback(this, 1);
        
        X1_mod_count = X1.mod_count();//save: X1.mod_count
        X2_mod_count = X2.mod_count();//save: X2.mod_count
        Y = __forward__(X1.engine(), X1.c(), X2.c());//wait until the computation on [X1, X2] is end
        Y_mod_count = Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        last_need_grads1 = X1.need_grad();
        last_need_grads2 = X2.need_grad();
        if(trace1 != null) last_need_grads1 = (last_need_grads1 || trace1.need_grads());
        if(trace2 != null) last_need_grads2 = (last_need_grads2 || trace2.need_grads());
        boolean need_grads = (last_need_grads1 || last_need_grads2 || ut.need_grads());
        Y.setTrace(this, 0, need_grads);
        
        return new Tensor[]{ Y };
    }

    @Override
    protected void traceBack(UnitCore<?> next, int out_index, int next_in_index) {
        Object value = arc.get(next);
        if (value == null) arc.put(next, next_in_index);
        else if (value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(2);
            indexes.add((Integer) value);
            indexes.add(next_in_index);
            arc.put(next, indexes);
        }
        else {//value instance of HashSet<Integer>
            HashSet<Integer> indexes = (HashSet<Integer>) value;
            indexes.add(next_in_index);
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override
    public Tensor[] collectGradientFromNext() {
        deltaY = aggregateGradient(arc); 
        arc.clear();
        return (deltaY == null? null : new Tensor[]{ deltaY });
    }

    protected abstract Tensor[] __backward__(Engine eg, Tensor deltaY,//compute deltaX1 and deltaX2
            boolean grad_inplace, boolean backward_grads,
            boolean backward_grads1, boolean backward_grads2);//mapping: X1, X2
    
    @Override
    public Tensor[] backward(Tensor... gradient) {//no gradient, no backward
        if(gradient == null) { deltaY = null; deltaX1 = null; deltaX2 = null; return null; }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = ut.backward_grads() && (last_need_grads1 || last_need_grads2);
        Tensor[] deltaX = __backward__(deltaY.engine(), deltaY.c(), 
                grad_inplace, backward_grads,
                last_need_grads1, last_need_grads2);
        
        if(deltaX != null) {//collect gradient for [X1, X2]
            deltaX1 = deltaX[0]; if(X1.need_grad()) X1.grad(deltaX1);
            deltaX2 = deltaX[1]; if(X2.need_grad()) X2.grad(deltaX2);
        }
        
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? deltaX : null);//deltaXarr.length == 2
    }

    @Override
    public Tensor gradient(int index) {
        if(index > 1 || index < 0) throw new IllegalArgumentException("tensor index out of range");
        return (index == 0 ? deltaX1 : deltaX2);
    }
    //</editor-fold>
}
