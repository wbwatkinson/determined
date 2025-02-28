import { Tooltip } from 'antd';
import React from 'react';
import TimeAgo from 'timeago-react';

import Avatar from 'components/Avatar';
import Badge, { BadgeType } from 'components/Badge';
import HumanReadableFloat from 'components/HumanReadableFloat';
import Icon from 'components/Icon';
import ProgressBar from 'components/ProgressBar';
import { paths } from 'routes/utils';
import {
  CheckpointState,
  CommandState, CommandTask, CommandType, ExperimentItem,
  Pagination, RunState, StartEndTimes, TrialItem,
} from 'types';
import { ConditionalWrapper } from 'utils/react';
import { canBeOpened } from 'utils/task';
import { getDuration, shortEnglishHumannizer } from 'utils/time';
import { commandTypeToLabel } from 'utils/types';
import { waitPageUrl } from 'wait';

import Link from './Link';
import css from './Table.module.scss';

type TableRecord = CommandTask | ExperimentItem | TrialItem;

export interface TablePaginationConfig {
  current: number;
  defaultPageSize: number;
  hideOnSinglePage: boolean;
  pageSize: number;
  showSizeChanger: boolean;
  total: number;
}

export type Renderer<T = unknown> = (text: string, record: T, index: number) => React.ReactNode;

export type GenericRenderer = <T extends TableRecord>(
  text: string, record: T, index: number,
) => React.ReactNode;

export type ExperimentRenderer = (
  text: string,
  record: ExperimentItem,
  index: number,
) => React.ReactNode;

export type TaskRenderer = (text: string, record: CommandTask, index: number) => React.ReactNode;

export const MINIMUM_PAGE_SIZE = 10;

export const defaultPaginationConfig = {
  current: 1,
  defaultPageSize: MINIMUM_PAGE_SIZE,
  pageSize: MINIMUM_PAGE_SIZE,
  showSizeChanger: true,
};

/* Table Column Renderers */

export const archivedRenderer = (archived: boolean): React.ReactNode => {
  return archived ? <Icon name="checkmark" /> : null;
};

export const durationRenderer = (times: StartEndTimes): React.ReactNode => {
  return shortEnglishHumannizer(getDuration(times));
};

export const humanReadableFloatRenderer = (num: number): React.ReactNode => {
  return <HumanReadableFloat num={num} />;
};

export const relativeTimeRenderer = (date: Date): React.ReactNode => {
  return (
    <Tooltip title={date.toLocaleString()}>
      <TimeAgo datetime={date} />
    </Tooltip>
  );
};

export const stateRenderer: Renderer<{ state: CommandState | RunState | CheckpointState }> =
(_, record) => (
  <div className={css.centerVertically}>
    <Badge state={record.state} type={BadgeType.State} />
  </div>
);

export const tooltipRenderer: Renderer = text => (
  <Tooltip placement="topLeft" title={text}><span>{text}</span></Tooltip>
);

export const userRenderer: Renderer<{ username: string }> = (_, record) => (
  <Avatar name={record.username} />
);

/* Command Task Table Column Renderers */

export const taskIdRenderer: TaskRenderer = (_, record) => (
  <Tooltip placement="topLeft" title={record.id}>
    <div className={css.centerVertically}>
      <ConditionalWrapper
        condition={canBeOpened(record)}
        wrapper={children => (
          <Link path={waitPageUrl(record)}>
            {children}
          </Link>
        )}>
        <Badge type={BadgeType.Id}>{record.id.split('-')[0]}</Badge>
      </ConditionalWrapper>
    </div>
  </Tooltip>
);

export const taskTypeRenderer: TaskRenderer = (_, record) => (
  <Tooltip placement="topLeft" title={commandTypeToLabel[record.type as unknown as CommandType]}>
    <div className={css.centerVertically}>
      <Icon name={record.type.toLowerCase()} />
    </div>
  </Tooltip>
);

export const taskNameRenderer: TaskRenderer = (id, record) => (
  <div>
    <ConditionalWrapper
      condition={canBeOpened(record)}
      wrapper={ch => (
        <Link path={waitPageUrl(record)}>
          {ch}
        </Link>
      )}>
      <span>{record.name}</span>
    </ConditionalWrapper>
  </div>
);

/* Experiment Table Column Renderers */

export const expermentDurationRenderer: ExperimentRenderer = (_, record) => {
  return shortEnglishHumannizer(getDuration(record));
};

export const experimentNameRenderer = (
  value: string | number | undefined,
  record: ExperimentItem,
): React.ReactNode => {
  return (
    <Link path={paths.experimentDetails(record.id)}>{value === undefined ? '' : value}</Link>
  );
};

export const experimentProgressRenderer: ExperimentRenderer = (_, record) => {
  return record.progress ? <ProgressBar
    percent={record.progress * 100}
    state={record.state} /> : null;
};

/* Table Helper Functions */

/*
 * For an `onClick` event on a table row, sometimes we have alternative and secondary
 * click interactions we want to capture. For example, we might want to capture different
 * link besides the one the table row is linked to. This function provides the means to
 * detect these alternative actions based on className definitions.
 */
export const isAlternativeAction = (event: React.MouseEvent): boolean => {
  const target = event.target as Element;
  if (target.className.includes('ant-checkbox-wrapper') ||
      target.className.includes('ignoreTableRowClick')) return true;
  return false;
};

/*
 * Default clickable row class name for Table components.
 */
export const defaultRowClassName = (options?: {
  clickable?: boolean,
  highlighted?: boolean,
}): string=> {
  const classes = [];
  if (options?.clickable) classes.push('clickable');
  if (options?.highlighted) classes.push('highlighted');
  return classes.join(' ');
};

export const getPaginationConfig = (
  count: number,
  pageSize?: number,
): Partial<TablePaginationConfig> => {
  return {
    defaultPageSize: MINIMUM_PAGE_SIZE,
    hideOnSinglePage: count < MINIMUM_PAGE_SIZE,
    pageSize,
    showSizeChanger: true,
  };
};

export const getFullPaginationConfig = (
  pagination: Pagination,
  total: number,
): TablePaginationConfig => {
  return {
    current: Math.floor(pagination.offset / pagination.limit) + 1,
    defaultPageSize: MINIMUM_PAGE_SIZE,
    hideOnSinglePage: total < MINIMUM_PAGE_SIZE,
    pageSize: pagination.limit,
    showSizeChanger: true,
    total,
  };
};
