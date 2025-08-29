//! # Event Replay - Event sourcing and replay functionality
//!
//! Provides event persistence and replay capabilities for system recovery
//! and audit purposes. Supports checkpoint-based replay for efficiency.

use crate::events::{Event, EventType, EventMetadata};
use crate::disruptor::RingBuffer;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write, BufRead};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Event journal for persistence
/// TODO: Add docs
pub struct EventJournal {
    /// Path to journal file
    path: PathBuf,
    /// Writer for appending events
    writer: Arc<RwLock<BufWriter<File>>>,
    /// Current segment number
    segment: usize,
    /// Events per segment
    segment_size: usize,
    /// Event count in current segment
    event_count: usize,
}

impl EventJournal {
    /// Create new journal
    pub fn new(base_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        
        let segment_path = path.join(format!("segment_{:06}.journal", 0));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&segment_path)?;
        
        Ok(Self {
            path,
            writer: Arc::new(RwLock::new(BufWriter::new(file))),
            segment: 0,
            segment_size: 100_000,  // 100k events per segment
            event_count: 0,
        })
    }
    
    /// Write event to journal
    pub fn write(&mut self, event: &Event) -> std::io::Result<()> {
        {
            let mut writer = self.writer.write().unwrap();
            
            // Serialize event to JSON
            let json = serde_json::to_string(event)?;
            writeln!(writer, "{}", json)?;
            writer.flush()?;
        } // Release writer lock here
        
        self.event_count += 1;
        
        // Rotate segment if needed
        if self.event_count >= self.segment_size {
            self.rotate_segment()?;
        }
        
        Ok(())
    }
    
    /// Rotate to new segment
    fn rotate_segment(&mut self) -> std::io::Result<()> {
        self.segment += 1;
        self.event_count = 0;
        
        let segment_path = self.path.join(format!("segment_{:06}.journal", self.segment));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&segment_path)?;
        
        self.writer = Arc::new(RwLock::new(BufWriter::new(file)));
        Ok(())
    }
}

/// Checkpoint for fast recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Sequence number
    pub sequence: u64,
    /// State snapshot (serialized)
    pub state: Vec<u8>,
    /// Events since last checkpoint
    pub event_count: usize,
}

/// Event replayer for recovery
/// TODO: Add docs
pub struct EventReplayer {
    /// Journal to read from
    journal_path: PathBuf,
    /// Checkpoints
    checkpoints: Vec<Checkpoint>,
    /// Replay buffer
    buffer: VecDeque<Event>,
    /// Current position
    position: u64,
}

impl EventReplayer {
    /// Create new replayer
    pub fn new(journal_path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            journal_path: journal_path.as_ref().to_path_buf(),
            checkpoints: Vec::new(),
            buffer: VecDeque::with_capacity(10_000),
            position: 0,
        })
    }
    
    /// Load events from journal
    pub fn load_events(&mut self, from_sequence: u64, to_sequence: Option<u64>) -> std::io::Result<Vec<Event>> {
        let mut events = Vec::new();
        
        // Find all journal segments
        let segments = self.find_segments()?;
        
        for segment_path in segments {
            let file = File::open(&segment_path)?;
            let reader = BufReader::new(file);
            
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                
                // Parse event
                let event: Event = serde_json::from_str(&line)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                
                // Filter by sequence range
                if event.metadata.sequence >= from_sequence {
                    if let Some(to) = to_sequence {
                        if event.metadata.sequence > to {
                            return Ok(events);
                        }
                    }
                    events.push(event);
                }
            }
        }
        
        Ok(events)
    }
    
    /// Find all segment files
    fn find_segments(&self) -> std::io::Result<Vec<PathBuf>> {
        let mut segments = Vec::new();
        
        for entry in std::fs::read_dir(&self.journal_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("journal") {
                segments.push(path);
            }
        }
        
        // Sort by segment number
        segments.sort();
        Ok(segments)
    }
    
    /// Replay events through handler
    pub async fn replay<F>(&mut self, handler: F, from: u64, to: Option<u64>) -> std::io::Result<usize>
    where
        F: Fn(&Event) -> Result<(), Box<dyn std::error::Error>>,
    {
        let events = self.load_events(from, to)?;
        let count = events.len();
        
        for event in events {
            handler(&event).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;
        }
        
        Ok(count)
    }
    
    /// Create checkpoint
    pub fn create_checkpoint(&self, sequence: u64, state: Vec<u8>) -> Checkpoint {
        Checkpoint {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            sequence,
            state,
            event_count: 0,
        }
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(&mut self, checkpoint: Checkpoint) -> std::io::Result<()> {
        self.checkpoints.push(checkpoint.clone());
        
        // Persist checkpoint to disk
        let checkpoint_path = self.journal_path.join("checkpoints.json");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&checkpoint_path)?;
        
        serde_json::to_writer(file, &self.checkpoints)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        Ok(())
    }
    
    /// Load checkpoints
    pub fn load_checkpoints(&mut self) -> std::io::Result<()> {
        let checkpoint_path = self.journal_path.join("checkpoints.json");
        
        if checkpoint_path.exists() {
            let file = File::open(&checkpoint_path)?;
            self.checkpoints = serde_json::from_reader(file)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        }
        
        Ok(())
    }
    
    /// Find nearest checkpoint
    pub fn find_nearest_checkpoint(&self, target_sequence: u64) -> Option<&Checkpoint> {
        self.checkpoints
            .iter()
            .rev()
            .find(|c| c.sequence <= target_sequence)
    }
}

/// Event store for querying historical events
/// TODO: Add docs
pub struct EventStore {
    /// Replayer for loading events
    replayer: EventReplayer,
    /// In-memory cache of recent events
    cache: Arc<RwLock<VecDeque<Event>>>,
    /// Maximum cache size
    cache_size: usize,
}

impl EventStore {
    /// Create new event store
    pub fn new(journal_path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            replayer: EventReplayer::new(journal_path)?,
            cache: Arc::new(RwLock::new(VecDeque::with_capacity(10_000))),
            cache_size: 10_000,
        })
    }
    
    /// Query events by time range
    pub fn query_by_time(&mut self, from: DateTime<Utc>, to: DateTime<Utc>) -> std::io::Result<Vec<Event>> {
        // Check cache first
        let cache = self.cache.read().unwrap();
        let mut results: Vec<Event> = cache
            .iter()
            .filter(|e| e.timestamp() >= from && e.timestamp() <= to)
            .cloned()
            .collect();
        drop(cache);
        
        // Load from journal if needed
        if results.is_empty() {
            let all_events = self.replayer.load_events(0, None)?;
            results = all_events
                .into_iter()
                .filter(|e| e.timestamp() >= from && e.timestamp() <= to)
                .collect();
        }
        
        Ok(results)
    }
    
    /// Query events by correlation ID
    pub fn query_by_correlation(&mut self, correlation_id: Uuid) -> std::io::Result<Vec<Event>> {
        // Check cache
        let cache = self.cache.read().unwrap();
        let mut results: Vec<Event> = cache
            .iter()
            .filter(|e| e.metadata.correlation_id == Some(correlation_id))
            .cloned()
            .collect();
        drop(cache);
        
        // Load from journal if needed
        if results.is_empty() {
            let all_events = self.replayer.load_events(0, None)?;
            results = all_events
                .into_iter()
                .filter(|e| e.metadata.correlation_id == Some(correlation_id))
                .collect();
        }
        
        Ok(results)
    }
    
    /// Add event to cache
    pub fn cache_event(&self, event: Event) {
        let mut cache = self.cache.write().unwrap();
        
        // Maintain cache size
        while cache.len() >= self.cache_size {
            cache.pop_front();
        }
        
        cache.push_back(event);
    }
}

/// Replay coordinator for system recovery
/// TODO: Add docs
pub struct ReplayCoordinator {
    /// Event store
    store: EventStore,
    /// Replayer
    replayer: EventReplayer,
    /// Target ring buffer
    ring_buffer: Arc<RingBuffer<Event>>,
}

impl ReplayCoordinator {
    /// Create new coordinator
    pub fn new(
        journal_path: impl AsRef<Path>,
        ring_buffer: Arc<RingBuffer<Event>>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            store: EventStore::new(journal_path.as_ref())?,
            replayer: EventReplayer::new(journal_path)?,
            ring_buffer,
        })
    }
    
    /// Replay from checkpoint
    pub async fn replay_from_checkpoint(&mut self, checkpoint_id: Uuid) -> std::io::Result<usize> {
        // Find checkpoint
        self.replayer.load_checkpoints()?;
        
        let checkpoint = self.replayer.checkpoints
            .iter()
            .find(|c| c.id == checkpoint_id)
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Checkpoint not found"
            ))?;
        
        // Replay events after checkpoint
        let count = self.replayer.replay(
            |event| {
                // Publish to ring buffer
                if let Some(sequence) = self.ring_buffer.next() {
                    self.ring_buffer.publish(sequence, event.clone());
                    Ok(())
                } else {
                    Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Ring buffer full or shutdown"
                    )) as Box<dyn std::error::Error>)
                }
            },
            checkpoint.sequence + 1,
            None,
        ).await?;
        
        Ok(count)
    }
    
    /// Replay time range
    pub async fn replay_time_range(
        &mut self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> std::io::Result<usize> {
        let events = self.store.query_by_time(from, to)?;
        let count = events.len();
        
        for event in events {
            if let Some(sequence) = self.ring_buffer.next() {
                self.ring_buffer.publish(sequence, event);
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Ring buffer full or shutdown"
                ));
            }
        }
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_event_journal() {
        let dir = tempdir().unwrap();
        let mut journal = EventJournal::new(dir.path()).unwrap();
        
        // Write events
        for i in 0..10 {
            let event = Event::new(
                EventType::HeartBeat {
                    timestamp: Utc::now(),
                    sequence: i,
                },
                "test".to_string(),
            );
            journal.write(&event).unwrap();
        }
        
        // Verify segment exists
        let segment_path = dir.path().join("segment_000000.journal");
        assert!(segment_path.exists());
    }
    
    #[tokio::test]
    async fn test_replay() {
        let dir = tempdir().unwrap();
        let mut journal = EventJournal::new(dir.path()).unwrap();
        
        // Write events
        let mut events = Vec::new();
        for i in 0..5 {
            let mut event = Event::new(
                EventType::HeartBeat {
                    timestamp: Utc::now(),
                    sequence: i,
                },
                "test".to_string(),
            );
            event.metadata.sequence = i;
            journal.write(&event).unwrap();
            events.push(event);
        }
        
        // Replay events
        let mut replayer = EventReplayer::new(dir.path()).unwrap();
        let loaded = replayer.load_events(0, Some(4)).unwrap();
        
        assert_eq!(loaded.len(), 5);
        assert_eq!(loaded[0].metadata.sequence, 0);
        assert_eq!(loaded[4].metadata.sequence, 4);
    }
}