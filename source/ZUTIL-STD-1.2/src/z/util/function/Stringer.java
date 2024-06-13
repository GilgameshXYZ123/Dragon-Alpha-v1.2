/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.function;

/**
 *
 * @author dell
 * @param <T>
 */
public interface Stringer<T> extends Processor<T, String>
{
    @Override
    public String process(T val);
}
