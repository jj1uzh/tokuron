CLASSES=NosyncCounterRunner.class SyncMethodCounterRunner.class SyncBlockCounterRunner.class

$(CLASSES): Main.java
	javac $^

run: $(CLASSES)
	java NosyncCounterRunner 1
	java NosyncCounterRunner 2
	java NosyncCounterRunner 3
	java NosyncCounterRunner 4
	java NosyncCounterRunner 10
	java NosyncCounterRunner 100
	java SyncMethodCounterRunner 1
	java SyncMethodCounterRunner 2
	java SyncMethodCounterRunner 3
	java SyncMethodCounterRunner 4
	java SyncMethodCounterRunner 10
	java SyncMethodCounterRunner 100
	java SyncBlockCounterRunner 1
	java SyncBlockCounterRunner 2
	java SyncBlockCounterRunner 3
	java SyncBlockCounterRunner 4
	java SyncBlockCounterRunner 10
	java SyncBlockCounterRunner 100

.PHONY: run
