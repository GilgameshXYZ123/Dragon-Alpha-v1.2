/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.Trace;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class SimpleCore<T extends SimpleUnit> extends UnitCore<T> {
    //<editor-fold defaultstate="collapsed" desc="member-parameters">
    transient private final UnitCoreMap<Object> arc = new UnitCoreMap<>();//solve the topology
    
    transient private Tensor X, deltaX;//input, input.grads
    transient private Tensor Y, deltaY;//output, output.grads
    transient private boolean last_need_grads;
    
    transient private int X_mod_count = -1;
    transient private int Y_mod_count = -1;
    //</editor-fold>

    public SimpleCore(T unit) { super(unit); }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final Tensor X() { return X; }
    public final Tensor Y() { return Y; }
    
    public final Tensor deltaX() { return deltaX; }
    public final Tensor deltaY() { return deltaY; }
    
    protected final Tensor holdX() { return (X.check()?  X.hold(X_mod_count, name() + ".X") : X); } 
    protected final Tensor holdY() { return (Y.check()?  Y.hold(Y_mod_count, name() + ".Y") : Y); }
    
    protected final boolean is_holdX() { return X.is_hold(X_mod_count); }
    protected final boolean is_holdY() { return Y.is_hold(Y_mod_count); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public Collection<UnitCore<?>> next() { return arc.keySet(); }

    @Override
    public void variables(TensorSet set) { 
        set.add(X, Y, deltaX, deltaY); 
    }
    
    @Override
    public void gc() {//the main part of gc will be down be unit (Module)
        if (X != null) { X.delete(); X = null; }
        if (Y != null) { Y.delete(); Y = null; }
        if (deltaX != null) { deltaX.delete(); deltaX = null; }
        if (deltaY != null) { deltaY.delete(); deltaY = null; }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected void __before_forward__(Engine eg, Tensor X) { }
    protected abstract Tensor __forward__(Engine eg, Tensor X);
    
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        arc.clear();//all one-off UnitCore references have been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X = input[0]; Trace trace = X.trace(); if(trace != null) trace.callback(this, 0);
      
        X_mod_count = X.mod_count();//save: X.mod_count
        __before_forward__(X.engine(), X);
        Y = __forward__(X.engine(), X.c());//wait until the computation on X is end
        Y_mod_count = Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        last_need_grads = X.need_grad();
        if (trace != null) last_need_grads = (last_need_grads || trace.need_grads());
        boolean need_grads = (last_need_grads || ut.need_grads());
        Y.setTrace(this, 0, need_grads);
     
        return new Tensor[]{ Y };
    }
       
    @Override
    protected synchronized void traceBack(UnitCore<?> next, int out_index, int next_in_index) {
        Object value = arc.get(next);
        if (value == null) arc.put(next, next_in_index);//create new arc for next node
        else if (value instanceof Integer) {//output[0] used twice for one specific next node
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
        return (deltaY == null ? null : new Tensor[]{ deltaY });
    }

    protected abstract Tensor __backward__(Engine eg, Tensor deltaY,//compute deltaX
            boolean grad_inplace, boolean backward_grads);
    
    @Override
    public Tensor[] backward(Tensor... gradient) {//no gradient, no backward
        if(gradient == null) { deltaY = null; deltaX = null; return null; }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = (ut.backward_grads() && last_need_grads);
        deltaX = __backward__(deltaY.engine(), deltaY.c(), grad_inplace, backward_grads);
        
        if(X.need_grad()) X.grad(deltaX);//collect gradient for X
        
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? new Tensor[]{ deltaX } : null);
    }

    @Override
    public Tensor gradient(int index) {
        if(index != 0) throw new IllegalArgumentException("tensor index out of range");
        return deltaX;
    }
    //</editor-fold>
}
