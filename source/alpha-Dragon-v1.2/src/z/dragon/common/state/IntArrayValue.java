/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.state;

import java.util.ArrayList;
import z.dragon.common.state.State.StateValue;

public final class IntArrayValue implements StateValue {
    private final int[] value;

    public IntArrayValue(int... value) {
        if(value == null) throw new RuntimeException();
        this.value = value;
    }

    @Override public int[] value() { return value; }

    @Override public Class<?> type() { return int[].class; }

    @Override
    public ArrayList<String> toStringLines() {
        StringBuilder sb = new StringBuilder(128);
        for (int v : value) sb.append(v).append(',');
        String line = sb.deleteCharAt(sb.length() - 1).toString();
        ArrayList<String> lines = new ArrayList<>(1);
        lines.add(line);
        return lines;
    }
}
