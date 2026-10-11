# Learned CPTs / priors (v3, fitted on all 1,000 users)

## Evidence priors P(flag=1) and rule

| Node | P(1) | Rule | TPR attack | FPR normal |
|---|---|---|---|---|
| AfterHoursLogin | 0.0245 | z>0.81 | 0.086 | 0.0243 |
| UnusualLoginFreq | 0.0472 | z>0.00 | 0.106 | 0.047 |
| WeekendLogin | 0.0003 | q 0.9995 | 0.0 | 0.0003 |
| HighMachineCount | 0.0002 | q 0.995 | 0.006 | 0.0002 |
| DeviceConnectActivity | 0.0038 | zq OR ctx>=2/14d | 0.267 | 0.003 |
| DeviceAfterHours | 0.0469 | z>0.00 | 0.361 | 0.046 |
| FileAccessCount | 0.0505 | z>0.61 | 0.323 | 0.0497 |
| FileCopyToRemovable | 0.0499 | z>0.71 | 0.379 | 0.0489 |
| ArchiveCreation | 0.0055 | rare 0.1 | 0.072 | 0.0053 |
| ExeActivity | 0.0015 | rare 0.02 | 0.021 | 0.0014 |
| UnusualFileTypes | 0.0000 | q 0.9995 | 0.0 | 0.0 |
| ExternalEmail | 0.0357 | zq OR ctx>=1/14d | 0.143 | 0.0354 |
| UnusualEmailVolume | 0.0007 | zq OR ctx>=1/30d | 0.217 | 0.0001 |
| AfterHoursEmail | 0.0005 | q 0.9995 | 0.0 | 0.0005 |
| ExternalLargeEmail | 0.0000 | q 0.9995 | 0.0 | 0.0 |
| UnusualWebActivity | 0.0022 | z>0.00 | 0.743 | 0.0001 |
| AfterHoursWeb | 0.0501 | z>1.86 | 0.107 | 0.0499 |
| CloudStorageAccess | 0.0070 | rare 0.05 | 0.007 | 0.007 |
| JobSearchActivity | 0.0367 | rare OR ctx>=5/14d | 0.325 | 0.0358 |
| HighDataMovement | 0.0184 | zq OR ctx>=3/14d | 0.208 | 0.0179 |
| HighFileCopyActivity | 0.0253 | zq OR ctx>=2/14d | 0.258 | 0.0246 |

## Noisy-OR nodes

| Node | Parents -> weights | Leak |
|---|---|---|
| AuthAnomaly | AfterHoursLogin: 0.09, UnusualLoginFreq: 0.05, HighMachineCount: 0.84 | 0.002 |
| DataAccessAnomaly | FileAccessCount: 0.00, FileCopyToRemovable: 0.00, HighFileCopyActivity: 0.00 | 0.000 |
| DataMovementAnomaly | FileCopyToRemovable: 0.10, HighDataMovement: 0.19, HighFileCopyActivity: 0.22 | 0.003 |
| CommunicationAnomaly | ExternalEmail: 0.05, UnusualEmailVolume: 0.62 | 0.002 |
| RemovableMediaAnomaly | DeviceConnectActivity: 0.89, DeviceAfterHours: 0.01, FileCopyToRemovable: 0.01 | 0.001 |
| TimeBasedAnomaly | WeekendLogin: 0.85, AfterHoursEmail: 0.79, AfterHoursWeb: 0.12, DeviceAfterHours: 0.15 | 0.004 |
| StagingAnomaly | ArchiveCreation: 0.54, ExeActivity: 0.83, UnusualFileTypes: 0.95 | 0.003 |
| FlightRiskAnomaly | JobSearchActivity: 0.71, ExternalEmail: 0.39, UnusualWebActivity: 0.50 | 0.007 |
| ShadowITAnomaly | CloudStorageAccess: 0.01, ExternalLargeEmail: 0.80, UnusualWebActivity: 0.90 | 0.000 |
| DataExfiltration | DataAccessAnomaly: 0.41, DataMovementAnomaly: 0.01, CommunicationAnomaly: 0.02, RemovableMediaAnomaly: 0.05, TimeBasedAnomaly: 0.00 | 0.000 |
| UnauthorizedAccess | AuthAnomaly: 0.01, DataAccessAnomaly: 0.81, TimeBasedAnomaly: 0.00 | 0.000 |
| ITSabotage | StagingAnomaly: 0.01, DataAccessAnomaly: 0.86, AuthAnomaly: 0.01, CommunicationAnomaly: 0.01 | 0.000 |
| IPTheft | FlightRiskAnomaly: 0.00, ShadowITAnomaly: 0.95, DataMovementAnomaly: 0.01, RemovableMediaAnomaly: 0.13 | 0.000 |

## RiskLevel CPT

| DataExfiltration | UnauthorizedAccess | ITSabotage | IPTheft | P(Low) | P(Med) | P(High) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 1.000 | 0.000 | 0.000 |
| 0 | 0 | 0 | 1 | 0.001 | 0.001 | 0.999 |
| 0 | 0 | 1 | 0 | 0.061 | 0.113 | 0.826 |
| 0 | 0 | 1 | 1 | 0.052 | 0.078 | 0.870 |
| 0 | 1 | 0 | 0 | 0.061 | 0.113 | 0.826 |
| 0 | 1 | 0 | 1 | 0.052 | 0.078 | 0.870 |
| 0 | 1 | 1 | 0 | 0.023 | 0.023 | 0.953 |
| 0 | 1 | 1 | 1 | 0.052 | 0.078 | 0.870 |
| 1 | 0 | 0 | 0 | 0.006 | 0.006 | 0.988 |
| 1 | 0 | 0 | 1 | 0.052 | 0.078 | 0.870 |
| 1 | 0 | 1 | 0 | 0.052 | 0.078 | 0.870 |
| 1 | 0 | 1 | 1 | 0.052 | 0.078 | 0.870 |
| 1 | 1 | 0 | 0 | 0.052 | 0.078 | 0.870 |
| 1 | 1 | 0 | 1 | 0.052 | 0.078 | 0.870 |
| 1 | 1 | 1 | 0 | 0.052 | 0.078 | 0.870 |
| 1 | 1 | 1 | 1 | 0.052 | 0.078 | 0.870 |