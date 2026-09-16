# Conditional Probability Tables (Expert Parameters)

These tables represent the expert-defined parameters for the 35-node CERT Bayesian Network.

## Layer 1: Observable Evidence (Priors)

| Node | P(Active=1) |
|---|---|
| AfterHoursLogin | 0.15 |
| UnusualLoginFreq | 0.12 |
| WeekendLogin | 0.05 |
| HighMachineCount | 0.08 |
| DeviceConnectActivity | 0.10 |
| DeviceAfterHours | 0.05 |
| FileAccessCount | 0.18 |
| FileCopyToRemovable | 0.08 |
| ArchiveCreation | 0.12 |
| ExeActivity | 0.07 |
| UnusualFileTypes | 0.05 |
| ExternalEmail | 0.20 |
| UnusualEmailVolume | 0.10 |
| AfterHoursEmail | 0.10 |
| ExternalLargeEmail | 0.05 |
| UnusualWebActivity | 0.14 |
| AfterHoursWeb | 0.10 |
| CloudStorageAccess | 0.08 |
| JobSearchActivity | 0.05 |
| HighDataMovement | 0.08 |
| HighFileCopyActivity | 0.05 |

## Layer 2: Behavioral Indicators (Noisy-OR)

| Node | Parents | Weights | Leak |
|---|---|---|---|
| AuthAnomaly | AfterHoursLogin, UnusualLoginFreq, HighMachineCount | 0.80, 0.70, 0.85 | 0.03 |
| DataAccessAnomaly | FileAccessCount, FileCopyToRemovable, HighFileCopyActivity | 0.60, 0.85, 0.90 | 0.04 |
| DataMovementAnomaly | FileCopyToRemovable, HighDataMovement, HighFileCopyActivity | 0.80, 0.75, 0.85 | 0.03 |
| CommunicationAnomaly | ExternalEmail, UnusualEmailVolume | 0.75, 0.70 | 0.05 |
| RemovableMediaAnomaly | DeviceConnectActivity, FileCopyToRemovable | 0.70, 0.90 | 0.02 |
| TimeBasedAnomaly | WeekendLogin, AfterHoursEmail, AfterHoursWeb, DeviceAfterHours | 0.85, 0.80, 0.80, 0.90 | 0.02 |
| StagingAnomaly | ArchiveCreation, ExeActivity, UnusualFileTypes | 0.80, 0.85, 0.95 | 0.03 |
| FlightRiskAnomaly | JobSearchActivity, ExternalEmail | 0.90, 0.70 | 0.02 |
| ShadowITAnomaly | CloudStorageAccess, ExternalLargeEmail | 0.85, 0.80 | 0.03 |

## Layer 3: Threat Hypotheses (Noisy-OR)

| Node | Parents | Weights | Leak |
|---|---|---|---|
| DataExfiltration | DataAccessAnomaly, DataMovementAnomaly, CommunicationAnomaly, RemovableMediaAnomaly | 0.85, 0.90, 0.70, 0.92 | 0.02 |
| UnauthorizedAccess | AuthAnomaly, DataAccessAnomaly | 0.88, 0.75 | 0.02 |
| ITSabotage | StagingAnomaly, DataAccessAnomaly, AuthAnomaly | 0.90, 0.85, 0.80 | 0.02 |
| IPTheft | FlightRiskAnomaly, ShadowITAnomaly, DataMovementAnomaly | 0.85, 0.90, 0.88 | 0.02 |

## Layer 4: Risk Level (Deterministic Mapping)

| DataExfiltration | UnauthorizedAccess | ITSabotage | IPTheft | P(Low) | P(Medium) | P(High) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.88 | 0.10 | 0.02 |
| 0 | 0 | 0 | 1 | 0.05 | 0.15 | 0.80 |
| 0 | 0 | 1 | 0 | 0.10 | 0.25 | 0.65 |
| 0 | 1 | 0 | 0 | 0.10 | 0.25 | 0.65 |
| 1 | 0 | 0 | 0 | 0.05 | 0.15 | 0.80 |
| 0 | 0 | 1 | 1 | 0.02 | 0.08 | 0.90 |
| 0 | 1 | 0 | 1 | 0.02 | 0.08 | 0.90 |
| 0 | 1 | 1 | 0 | 0.02 | 0.08 | 0.90 |
| 1 | 0 | 0 | 1 | 0.02 | 0.08 | 0.90 |
| 1 | 0 | 1 | 0 | 0.02 | 0.08 | 0.90 |
| 1 | 1 | 0 | 0 | 0.02 | 0.08 | 0.90 |
| 0 | 1 | 1 | 1 | 0.02 | 0.08 | 0.90 |
| 1 | 0 | 1 | 1 | 0.02 | 0.08 | 0.90 |
| 1 | 1 | 0 | 1 | 0.02 | 0.08 | 0.90 |
| 1 | 1 | 1 | 0 | 0.02 | 0.08 | 0.90 |
| 1 | 1 | 1 | 1 | 0.02 | 0.08 | 0.90 |
