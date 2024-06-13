/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.function;

/**
 *
 * @author dell
 * @param <A>
 * @param <B>
 * @param <C>
 */
public interface TriConsumer <A,B,C>
{
    public void accept(A arg1, B arg2, C arg3);
}
