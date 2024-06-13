/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.exception;

/**
 *
 * @author dell
 */
public class CheckException extends RuntimeException
{
    public static final String MSG="PoolClearException: ";
    
    public CheckException()
    {
        super(MSG);
    }
    public CheckException(String msg)
    {
        super(MSG+msg);
    }
    public CheckException(Throwable e)
    {
        super(e);
    }
}