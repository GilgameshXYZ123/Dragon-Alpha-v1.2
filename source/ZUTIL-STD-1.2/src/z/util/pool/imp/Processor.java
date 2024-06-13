/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.imp;

import java.util.concurrent.Future;

/**
 *
 * @author dell
 * @param <R>
 * @param <T>
 */
public interface Processor<R,T>
{
    public void process(R result, T resource) throws Exception;
}
