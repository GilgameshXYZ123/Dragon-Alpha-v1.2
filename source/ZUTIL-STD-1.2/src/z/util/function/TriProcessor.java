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
 * @param <R>
 */
public interface TriProcessor<A, B, C, R> 
{
    public R process(A args1, B args2, C args3) throws Exception;
}
