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
 * S' = S / (1 - Us)
 * W = W - lr * deltaW / [sqrt(S') + eps]
 * W = W - lr * deltaW / [sqrt(S/(1-Us)) + eps]
 * W = W - [lr * sqrt(1 - Us)] / [sqrt(S) + sqrt(1 - Us)*eps)]
 * let: lr_t = lr * sqrt(1 - Us)
 *       eps_t = eps * sqrt(1 - Us) 
 * W = W - lr_t * deltaW / [sqrt(S) + eps_t]
 * @author Gilgamesh
 */
public class RMSprop extends Optimizer
{
    protected float lr_t, eps_t;
    
    protected float beta, b1, b2, eps, expBeta;
    protected Tensor[] S;
    
    protected float L1 = 0.0f;
    protected float L2 = 0.0f;
    
    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta, float eps)  {
        this.beta = beta; this.eps = eps;
        S = Tensor.zero_like(params); expBeta = 1.0f;
        Tensor.sync(S);
    }
    //</editor-fold>
    public RMSprop(Parameter[] params, float lr, float beta, float eps) {
        super(params, lr);
        __init__(beta, eps);
    }
    
    public RMSprop(Collection<Parameter> params, float lr, float beta, float eps) {
        super(params, lr);
        __init__(beta, eps);
    }
    
    public RMSprop(Map<String, Parameter> paramMap, float lr, float beta, float eps) {
        super(paramMap, lr);
        __init__(beta, eps);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public RMSprop learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg_sq() { return S; }
    
    public float beta() {return beta;}
    public RMSprop beta(float beta) { this.beta = beta; return this; }
    
    public float eps() {return eps;}
    public RMSprop eps(float eps) { this.eps = eps; return this; }
    
    public float L1() { return L1; }
    public RMSprop L1(float L1) { this.L1 = L1; return this; }
    
    public float L2() { return L2; }
    public RMSprop L2(float L2) { this.L2 = L2; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr);
        sb.append(", [beta, eps] = ")
                .append(beta).append(", ")
                .append(eps).append(")");
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
        StateValue expBetas = dic.get("expBetas");
        State.set(expBetas, "fail to load expBetas", partial, ()->{
            expBeta = Vector.to_float_vector(expBetas.toStringLines(), 1)[0];
        });
    }

    //param-state---------------------------------------------------------------
    protected String exp_avg_sq_key(String param_name) { return param_name + ".exp_avg_sq"; }
    
    @Override
    protected void param_state(State dic, int index, String paramN_name) {
        dic.put(exp_avg_sq_key(paramN_name), S[index]);
    }

    @Override
    protected void update_param_state(State dic, boolean partial, int index, String paramName) {
        String exp_avg_sq_key = exp_avg_sq_key(paramName);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: updates">
    @Override
    protected void __before_update__() {
        expBeta *= beta;//init: expBeta = 1
        
        b1 = beta; b2 = 1 - beta; //exp_avg_sq
        
        double correct_beta =  Math.sqrt(1 - expBeta);
        lr_t = (float) (lr * correct_beta);
        eps_t = (float) (eps * correct_beta);
    }

    @Override
    protected void __update__(int index, Tensor grad, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor s = S[index];//exp_avg_sq 
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.rmsprop(w, s, b1, b2, eps_t, grad, lr_t, L1, L2);
        else eg.rmsprop(w, s, b1, b2, eps_t, grad, lr_t);
    }

    @Override
    protected void __update__(int index, Collection<Tensor> grads, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor s = S[index];//exp_avg_sq 
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.rmsprop(w, s, b1, b2, eps_t, grads, lr_t, L1, L2);
        else eg.rmsprop(w, s, b1, b2, eps_t, grads, lr_t);
    }
    
    @Override
    protected void __clear__() {
        if(S != null) { Tensor.delete(S); S = null; }
    }
    //</editor-fold>
}
