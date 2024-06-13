/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public interface Hook<T extends UnitCore> {
    public void callback(T self);
}
