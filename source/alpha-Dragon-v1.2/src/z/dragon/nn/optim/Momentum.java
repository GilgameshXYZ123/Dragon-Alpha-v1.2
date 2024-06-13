/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Momentum extends Optimizer
{
    protected float lr_t;
    
    protected float beta, a1, a2, expBeta;
    protected Tensor[] V;
    
    protected float L1 = 0.0f;
    protected float L2 = 0.0f;

    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta) {
        this.beta = beta;
        V = Tensor.zero_like(params); expBeta = 1.0f;
        Tensor.sync(V);
    }
    //</editor-fold>
    public Momentum(Parameter[] params, float lr, float beta) {
        super(params, lr);
        __init__(beta);
    }
    
    public Momentum(Collection<Parameter> params, float lr, float beta) {
        super(params, lr);
       __init__(beta);
    }
    
    public Momentum(Map<String, Parameter> paramMap, float lr, float beta) {
        super(paramMap, lr);
       __init__(beta);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Momentum learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    
    public float beta() { return beta; }
    public Momentum beta(float beta) { this.beta = beta; return this; }
    
    public float L1() { return L1; }
    public Momentum L1(float L1) { this.L1 = L1; return this; }
    
    public float L2() { return L2; }
    public Momentum L2(float L2) { this.L2 = L2; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{  learning_rate = ").append(lr);
        sb.append(", beta = ").append(beta);
        sb.append(", [L1, L2] = [")
                .append(L1).append(", ")
                .append(L2).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hypher-state--------------------------------------------------------------
    @Override
    protected void hypher_state(State dic) { 
        dic.put("expBetas", State.floats(expBeta)); 
    }
    
    @Override
    protected void update_hypher_state(State dic, boolean partial) {
        StateValue expBetas = dic.get("expBeta");
        State.set(expBetas, "fail to load expBeta", partial, ()->{
            expBeta = Vector.to_float_vector(expBetas.toStringLines(), 1)[0];
        });
    }
    
    //param-state---------------------------------------------------------------
    protected String exp_avg_key(String param_name) { return param_name + ".exp_avg"; }
    
    @Override 
    protected void param_state(State dic, int index, String param_name) {
        dic.put(exp_avg_key(param_name), V[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String exp_avg_key = exp_avg_key(param_name);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load: " + exp_avg_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {
        expBeta *= beta;//init: expBeta = 1
        a1 = beta; a2 = 1 - beta;//exp_avg
        lr_t = (float) (lr / (1.0 - expBeta));
    }
    
    @Override
    protected void __update__(int index, Tensor grad, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v =  V[index];//exp_avg
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.momentum(w, v, a1, a2, grad, lr_t, L1, L2); 
        else eg.momentum(w, v, a1, a2, grad, lr_t);
    }
    
    @Override
    protected void __update__(int index, Collection<Tensor> grads, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v =  V[index];//exp_avg
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.momentum(w, v, a1, a2, grads, lr_t, L1, L2);
        else eg.momentum(w, v, a1, a2, grads, lr_t);
    }
    
    @Override
    protected void __clear__() { 
        if(V != null) { Tensor.delete(V); V = null; }
    }
    //</editor-fold>
}
