import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

interface Counter {
    int get();
    void incr();
}

class NosyncCounter implements Counter {
    private int cnt = 0;
    public int get() { return cnt; }
    public void incr() { cnt += 1; }
}

class SyncMethodCounter implements Counter {
    private int cnt = 0;
    public synchronized int get() { return cnt; }
    public synchronized void incr() { cnt += 1; }
}

class SyncBlockCounter implements Counter {
    private int cnt = 0;
    private Object lock = new Object();
    public int get() { synchronized(lock) { return cnt; } }
    public void incr() { synchronized(lock) { cnt += 1; } }
}

class Timer {
    long start = 0;
    Timer(long start) {
        this.start = start;
    }
    public void end(long t) {
        System.out.printf("%9.4f\n", (t - start) / 1000.0);
    }
}

class Run implements Callable<Void> {
    Counter counter;
    Timer timer;
    int getValue = 0;

    Run(Counter counter, Timer t) {
        this.counter = counter;
        this.timer = t;
    }

    public Void call() {
        for (int i = 0; i < 1000000; i++) {
            getValue = counter.get();
            counter.incr();
        }
        long time = System.currentTimeMillis();
        timer.end(time);
        return null;
    }
}

class Runner {
    Counter counter;
    int threadsNum;

    void run() {
        ExecutorService executor = Executors.newFixedThreadPool(threadsNum);
        ArrayList<Run> runs = new ArrayList<Run>(threadsNum);
        for (int i = 0; i < threadsNum; i++) {
            runs.add(new Run(counter, new Timer(System.currentTimeMillis())));
        }
        try {
            executor.invokeAll(runs);
            executor.shutdown();
        } catch (Exception e) {}
    }
}

class NosyncCounterRunner extends Runner {
    NosyncCounterRunner(int threadsNum) {
        this.counter = new NosyncCounter();
        this.threadsNum = threadsNum;
    }

    public static void main(String[] args) {
        new NosyncCounterRunner(Integer.parseInt(args[0])).run();
    }
}

class SyncMethodCounterRunner extends Runner {
    SyncMethodCounterRunner(int threadsNum) {
        this.counter = new SyncMethodCounter();
        this.threadsNum = threadsNum;
    }

    public static void main(String[] args) {
        new SyncMethodCounterRunner(Integer.parseInt(args[0])).run();
    }
}

class SyncBlockCounterRunner extends Runner {
    SyncBlockCounterRunner(int threadsNum) {
        this.counter = new SyncBlockCounter();
        this.threadsNum = threadsNum;
    }

    public static void main(String[] args) {
        new SyncBlockCounterRunner(Integer.parseInt(args[0])).run();
    }
}
