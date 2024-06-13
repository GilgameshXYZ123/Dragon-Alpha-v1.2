/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.function;

/**
 *
 * @author dell
 * @param <A> the type of the first argument
 * @param <B> the type of the second argument
 * @param <R> the type of the returned value
 */
public interface BiProcessor<A, B, R>
{
    public R process(A args1, B args2) throws Exception;
}
