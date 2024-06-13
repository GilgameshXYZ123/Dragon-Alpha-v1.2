/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math1;

import z.dragon.nn.core.simple.math1.CoreGelu;
import z.dragon.nn.unit.simple.SimpleFunction;

/**
 *
 * @author Gilgamesh
 */
public class Gelu extends SimpleFunction 
{
    @Override
    protected CoreGelu<?> create_unit_core() {
        return new CoreGelu<>(this);
    }
}
