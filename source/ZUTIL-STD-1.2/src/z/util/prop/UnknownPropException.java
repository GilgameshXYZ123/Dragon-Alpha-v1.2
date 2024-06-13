/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.prop;

/**
 *
 * @author dell
 */
public class UnknownPropException extends RuntimeException
{
    public static final String MESSAGE="UnknownPropException";
    public UnknownPropException()
    {
        super();
    }
    public UnknownPropException(String msg)
    {
        super(MESSAGE+msg);
    }
}
