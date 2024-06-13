/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core;

/**
 *
 * @author Gilgamesh
 */
public interface BackwardHookable
{
    
    public BackwardHookable hook_before_backward(Hook hook);
    public BackwardHookable hook_after_backward(Hook hook);
    
    default BackwardHookable remove_before_backward() { hook_before_backward(null); return this; }
    default BackwardHookable remove_after_backward() { hook_after_backward(null);return this;
    }
}
