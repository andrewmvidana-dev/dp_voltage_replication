# Feeder candidates for the BNP viability regime

The acceptance table in [bnp_viability_target.md](bnp_viability_target.md) was
written before this screening. No candidate was compiled, simulated, or fitted.
Existing measurements were neither rerun nor changed. Counts below distinguish
published network statistics from static counts of public model definitions.

## Three candidate families, including both EPRI circuits

Node counts are published counts for the cited versions, not local measurements.
Classes are the pipeline's three demand-ranked groups, not measured residential,
commercial, and industrial customer populations.

| Candidate | Published nodes | Customers and represented loads | Records per class at 90 days | Free OpenDSS availability |
|---|---:|---|---|---|
| IEEE 8500, balanced | 8,531 in GridAPPS-D [1] | 1,177 Load objects [2]; groups 393/392/392. Physical customers per group unverified. | 35,370 / 35,280 / 35,280 | Yes: balanced Master.dss and its dependencies [3]. |
| EPRI Ckt5 | 3,437 [4] | EPRI reports 1,379 residential loads; 1,379 Load declarations [2]. Groups 460/460/459. Customer-to-object mapping still needs validation. | 41,400 / 41,400 / 41,310 | Yes: Master_ckt5.dss and dependencies [5]. |
| EPRI Ckt7 | 2,452 in the cited study [6] | Published 5,694 customers [6], but 867 downstream Load objects plus 39 upstream feeder equivalents [7]. Current enumeration gives groups 302/302/302; physical customers per group unknown. | 27,180 each, including equivalents | Yes: Master_ckt7.dss and dependencies [7]. |
| PNNL taxonomy: R2_12_47_2 | 1,631 in GridAPPS-D [1] | Source advertises approximately 4,000 houses, not a verified count of independent records. Inspected CIM contains 214 EnergyConsumer objects; customer counts by pipeline class unknown. | Not established for an OpenDSS export | Free CIM/GridLAB-D source and OpenDSS export tools [8]; ready-to-run DSS file not verified in the inspected revision. |

The IEEE 8500 unbalanced variant was also inspected: 2,354 Load declarations,
giving groups 785/785/784 and 70,650/70,650/70,560 object-days at 90 days [2,3].
Those split secondary legs do not establish twice as many independent customers.
It is reported here as a model variant, not used to inflate the recommendation.

## Screening results, including failures

For C=6 and the fixed reference signal s=0.0154, the target m per class is
134,965 / 53,986 / 26,993 at delta=0.02 / 0.05 / 0.10.
Required days below use the smallest represented class and round upward.
They describe record counts, not demonstrated utility or available meter history.

| Candidate | Required days: delta 0.02 / 0.05 / 0.10 | At 90 days, C=6 |
|---|---|---|
| IEEE 8500 balanced | 345 / 138 / 69 | Fails 0.02 and 0.05; passes 0.10 count screen. |
| IEEE 8500 unbalanced | 173 / 69 / 35 | Fails 0.02; passes 0.05 and 0.10 for object-days only. |
| EPRI Ckt5 | 295 / 118 / 59 | Fails 0.02 and 0.05; passes 0.10 count screen. |
| EPRI Ckt7, all 906 objects | 447 / 179 / 90 | Fails 0.02 and 0.05; barely passes 0.10, including upstream equivalents. |
| PNNL R2_12_47_2 | Not established | Not qualified: individual OpenDSS record count is unverified. |

Ckt7's 867 downstream objects alone would form 289-member groups and require
468 / 187 / 94 days. The advertised 5,694 customers cannot be substituted for
either object count. If the 214 PNNL CIM consumers mapped one-to-one to records,
the smallest group would have 71 members and require 1,901 / 761 / 381 days.
That mapping has not been validated. No other feeder was evaluated.

## Preprocessing transfer: conditional, not tested

The switch-merging matrix operation and stub Schur complement in network.py
are structurally reusable. None of these candidates is certified to transfer
unchanged. _switch_node_pairs currently accepts IsSwitch() or Length()<=1e-3
without checking terminal open state or converting length units. That needs
review on each model before merging. Stub elimination protects loads and sources;
other injections, neutral nodes, and isolated components need explicit review.

- IEEE 8500: check split-phase/neutral connections, regulators, and switches.
  Its larger dense admittance matrices also require a memory/runtime assessment.
- Ckt5: the best first transfer candidate by published size. Check secondary
  voltage bases, switch encoding, capacitor states, and zero-injection residuals.
- Ckt7: additionally distinguish detailed downstream loads from 39 feeder
  equivalents. The master applies load allocation factors, so ratings must be
  read after the intended allocation and solution procedure.
- PNNL: first validate CIM-to-DSS conversion, triplex connections and customer
  population. The published node count does not certify the converted network.

No numerical preprocessing changes are proposed in this cleanup.

## Recommendation

Use **EPRI Ckt5 with 365 historical days, T=96 and C=6**, initially screening
delta=0.02. The smallest class gives 459*365=167,535 records, above 134,965.
The implied pre-repair uniform sd is approximately 0.01241, below the reference
0.0154. These are analytical calculations, not measurements on Ckt5.

Ckt5 offers more directly represented loads and fewer published nodes than
balanced IEEE 8500. It avoids relying on Ckt7's customer-to-aggregate mismatch
or the unverified PNNL export. Keeping T and C fixed isolates the effect of
sample count. The existing synthetic-history generator can request 365 days;
availability of 365 days of individual metered histories is not established.

This recommendation supports a next experiment, not a claim of demonstrated
viability or superiority to Gaussian noise. Each new class's covariance scale,
R, clipping bias, temporal fidelity and voltage utility remain unmeasured.
No new feeder implementation or experiment is included at this stage.

## Sources and static-count provenance

[1] [GridAPPS-D model inventory](https://github.com/GRIDAPPSD/Powergrid-Models/blob/eaa0c3dcbc607e01c3f98092e3dea18929ae4d60/README.md).

[2] [DSS-Extensions Load definition inventory](https://dss-extensions.org/dss-format/Load.html).
Counts were independently checked from the named load files in the public
OpenDSS mirror at revision 5005c668a72d20775f4c2d060feebb2866ba1d38.
This was text inspection in memory, with no feeder files saved or executed.

[3] [IEEE 8500 OpenDSS files](https://github.com/tshort/OpenDSS/tree/5005c668a72d20775f4c2d060feebb2866ba1d38/Distrib/IEEETestCases/8500-Node).
The tshort repository is an unofficial mirror of EPRI's public distribution.

[4] [EPRI Ckt5 model description, Table 2-1](https://restservice.epri.com/publicdownload/000000003002015283/0/Product).

[5] [Ckt5 OpenDSS files](https://github.com/tshort/OpenDSS/tree/5005c668a72d20775f4c2d060feebb2866ba1d38/Distrib/EPRITestCircuits/ckt5).

[6] [Published test-system statistics, Table II](https://www.osti.gov/servlets/purl/1855974).

[7] [Ckt7 OpenDSS files](https://github.com/tshort/OpenDSS/tree/5005c668a72d20775f4c2d060feebb2866ba1d38/Distrib/EPRITestCircuits/ckt7).
Loads_ckt7.dss defines 867 loads; Substation_ckt7.dss defines 39 equivalents.

[8] [PNNL CIM source](https://github.com/GRIDAPPSD/Powergrid-Models/blob/eaa0c3dcbc607e01c3f98092e3dea18929ae4d60/models/feeders/CIM/XML/R2_12_47_2.xml)
and [CIMHub export capabilities](https://github.com/GRIDAPPSD/CIMHub/blob/master/PYPIDESC.md).
Static XML inspection counted 214 EnergyConsumer and 853 ConnectivityNode
elements. This is not a measurement of exported OpenDSS customer records.
