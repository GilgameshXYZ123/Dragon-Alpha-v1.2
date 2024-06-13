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
 * @param <R>
 */
public interface Processor<A, R>
{
    public R process(A arg1);
}
